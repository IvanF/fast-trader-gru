package grid

import (
	"math"

	"github.com/fast-trader-gru/oms_execution/internal/bybit"
	"github.com/fast-trader-gru/oms_execution/internal/liquidity"
	"github.com/fast-trader-gru/oms_execution/internal/models"
)

// ExitLevel is one reduce-only limit exit order.
type ExitLevel struct {
	Price float64
	Qty   float64
	Kind  string // fee_aware_tp | liquidity_tp | ml_tp | stop_loss
}

// ExitGrid holds SL + single TP for a filled position.
type ExitGrid struct {
	StopLoss    ExitLevel
	TakeProfits []ExitLevel
	SmartSL     bool
}

// atrSLMultiplier scales SL distance by regime: trending needs room, choppy tightens.
func atrSLMultiplier(regime string) float64 {
	switch regime {
	case "Trending":
		return 1.8
	case "Breakout":
		return 2.2
	default:
		return 1.2
	}
}

// computeSmartSL places SL at nearest S/R level, capped by max distance, tightened for knives.
func computeSmartSL(
	direction string,
	fillPrice float64,
	ob models.OrderbookSnapshot,
	signal models.TradeSignal,
	tickSize, risk, maxSLPct float64,
) (float64, bool) {
	var sl float64
	foundSR := false
	if direction == "LONG" {
		support := liquidity.FindNearestSupport(ob, fillPrice)
		if support.Price > 0 {
			sl = support.Price - tickSize*2
			foundSR = true
		} else {
			sl = fillPrice - risk
		}

		if signal.MacroTrend15m < -0.005 {
			trendTighten := math.Max(0.6, 1.0+signal.MacroTrend15m*10)
			sl = fillPrice - (fillPrice-sl)*trendTighten
		}

		maxSLDist := fillPrice * maxSLPct
		minSL := fillPrice - maxSLDist
		if sl < minSL {
			sl = minSL
		}

		if sl >= fillPrice {
			sl = fillPrice - risk
		}
	} else {
		resistance := liquidity.FindNearestResistance(ob, fillPrice)
		if resistance.Price > 0 {
			sl = resistance.Price + tickSize*2
			foundSR = true
		} else {
			sl = fillPrice + risk
		}

		if signal.MacroTrend15m > 0.005 {
			trendTighten := math.Max(0.6, 1.0-signal.MacroTrend15m*10)
			sl = fillPrice + (sl-fillPrice)*trendTighten
		}

		maxSLDist := fillPrice * maxSLPct
		maxSL := fillPrice + maxSLDist
		if sl > maxSL {
			sl = maxSL
		}

		if sl <= fillPrice {
			sl = fillPrice + risk
		}
	}

	return roundToTick(sl, tickSize), foundSR
}

// BuildExitGrid places SL + single TP covering 100% of position.
func BuildExitGrid(
	direction string,
	fillPrice, plannedEntry, plannedSL float64,
	ob models.OrderbookSnapshot,
	signal models.TradeSignal,
	tickSize, totalQty, qtyStep, minQty float64,
	opts ExitGridOptions,
) ExitGrid {
	vm := signal.VolatilityMultiplier
	if vm <= 0 {
		vm = 1.0
	}

	minSLPct := opts.MinSLPct
	maxSLPct := opts.MaxSLPct

	if vm > 0 && vm != 1.0 {
		slVolMult := math.Sqrt(vm)
		minSLPct *= slVolMult
		maxSLPct *= slVolMult
	}

	var slPrice, tpPrice float64
	smartSL := false
	tpKind := "fee_aware_tp"

	dynamicSLPct := signal.DynamicSLPct
	dynamicTPPct := signal.DynamicTPPct
	hasPythonTP := len(signal.TakeProfits) > 0

	// === SL computation ===
	if dynamicSLPct > 0 && dynamicTPPct > 0 {
		slPct := dynamicSLPct
		if slPct < minSLPct {
			slPct = minSLPct
		}
		if maxSLPct > 0 && slPct > maxSLPct*3 {
			slPct = maxSLPct * 3
		}
		if direction == "LONG" {
			slPrice = fillPrice * (1.0 - slPct)
		} else {
			slPrice = fillPrice * (1.0 + slPct)
		}
		slPrice = roundToTick(slPrice, tickSize)
	} else if signal.StopLoss > 0 {
		slPrice = roundToTick(signal.StopLoss, tickSize)
		if direction == "LONG" && slPrice >= fillPrice {
			slPrice = roundToTick(fillPrice*(1.0-minSLPct), tickSize)
		}
		if direction == "SHORT" && slPrice <= fillPrice {
			slPrice = roundToTick(fillPrice*(1.0+minSLPct), tickSize)
		}
		slDistPct := math.Abs(slPrice-fillPrice) / fillPrice
		if slDistPct < minSLPct {
			if direction == "LONG" {
				slPrice = roundToTick(fillPrice*(1.0-minSLPct), tickSize)
			} else {
				slPrice = roundToTick(fillPrice*(1.0+minSLPct), tickSize)
			}
		}
		if maxSLPct > 0 && slDistPct > maxSLPct {
			if direction == "LONG" {
				slPrice = roundToTick(fillPrice*(1.0-maxSLPct), tickSize)
			} else {
				slPrice = roundToTick(fillPrice*(1.0+maxSLPct), tickSize)
			}
		}
	} else {
		slResult := liquidity.ComputeLiquiditySL(
			direction, fillPrice, ob, tickSize, minSLPct, maxSLPct,
		)
		slPrice = slResult.Price
		smartSL = slResult.Source == "liquidity_zone" || slResult.Source == "nearest_level"
	}

	// === SL tick enforcement: minimum 5 ticks from entry ===
	minSLFromTicks := 5.0 * tickSize
	if direction == "LONG" {
		slMaxDist := fillPrice - minSLFromTicks
		if slPrice <= 0 || slPrice > slMaxDist {
			slPrice = roundToTick(slMaxDist, tickSize)
		}
	} else {
		slMinDist := fillPrice + minSLFromTicks
		if slPrice <= 0 || slPrice < slMinDist {
			slPrice = roundToTick(slMinDist, tickSize)
		}
	}

	// === TP computation ===
	maxTPDist := opts.MaxTPPct
	if maxTPDist <= 0 {
		maxTPDist = 0.015
	}
	// High-volatility: tighter TP for faster profit capture
	if vm > 1.5 {
		volTPScale := 1.5 / vm // vm=2.0 → scale=0.75, vm=3.0 → scale=0.5
		maxTPDist *= volTPScale
		if maxTPDist < opts.MinTPPct {
			maxTPDist = opts.MinTPPct
		}
	}

	// Priority 1: Liquidity wall (nearest support for SHORT, resistance for LONG)
	// This places TP at a real orderbook level where price is likely to bounce
	if tpPrice <= 0 {
		if direction == "SHORT" {
			supportWall := liquidity.FindNearestSupport(ob, fillPrice)
			if supportWall.Price > 0 {
				wallDist := math.Abs(fillPrice-supportWall.Price) / fillPrice
				feeMinDist := math.Max((opts.EntryFeeRate+opts.ExitFeeRate)*2, 0.003)
				if wallDist >= feeMinDist && wallDist <= maxTPDist {
					tpPrice = roundToTick(supportWall.Price+tickSize*2, tickSize)
					tpKind = "liquidity_tp"
				}
			}
		} else {
			resistanceWall := liquidity.FindNearestResistance(ob, fillPrice)
			if resistanceWall.Price > 0 {
				wallDist := math.Abs(resistanceWall.Price-fillPrice) / fillPrice
				feeMinDist := math.Max((opts.EntryFeeRate+opts.ExitFeeRate)*2, 0.003)
				if wallDist >= feeMinDist && wallDist <= maxTPDist {
					tpPrice = roundToTick(resistanceWall.Price-tickSize*2, tickSize)
					tpKind = "liquidity_tp"
				}
			}
		}
	}

	// Priority 2: Python-computed TP from spike-detection
	if tpPrice <= 0 && hasPythonTP {
		candidateTP := roundToTick(signal.TakeProfits[0], tickSize)
		if (direction == "LONG" && candidateTP > fillPrice) ||
			(direction == "SHORT" && candidateTP < fillPrice) {
			tpDist := math.Abs(candidateTP-fillPrice) / fillPrice
			if tpDist <= maxTPDist && tpDist > 0 {
				tpPrice = candidateTP
				tpKind = "ml_tp"
			}
		}
	}

	// Priority 3: fee-aware TP (formula fallback)
	if tpPrice <= 0 {
		tpPrice = CalculateExitPrice(fillPrice, direction, opts.EntryFeeRate, opts.ExitFeeRate, opts.TargetNetProfitPct, tickSize)
		if tpPrice <= 0 {
			fallbackDist := fillPrice * (opts.EntryFeeRate + opts.ExitFeeRate + 0.005)
			if direction == "LONG" {
				tpPrice = roundToTick(fillPrice+fallbackDist, tickSize)
			} else {
				tpPrice = roundToTick(fillPrice-fallbackDist, tickSize)
			}
		}
	}

	// Enforce MinTPPct only for fee-aware TPs
	if tpKind == "fee_aware_tp" {
		tpDist := math.Abs(tpPrice-fillPrice) / fillPrice
		if tpDist < opts.MinTPPct {
			if direction == "LONG" {
				tpPrice = roundToTick(fillPrice*(1.0+opts.MinTPPct), tickSize)
			} else {
				tpPrice = roundToTick(fillPrice*(1.0-opts.MinTPPct), tickSize)
			}
		}
	}

	// === TP tick enforcement: minimum 3 ticks from entry ===
	minTPFromTicks := 3.0 * tickSize
	if direction == "LONG" {
		tpMinDist := fillPrice + minTPFromTicks
		if tpPrice <= 0 || tpPrice < tpMinDist {
			tpPrice = roundToTick(tpMinDist, tickSize)
		}
	} else {
		tpMaxDist := fillPrice - minTPFromTicks
		if tpPrice <= 0 || tpPrice > tpMaxDist {
			tpPrice = roundToTick(tpMaxDist, tickSize)
		}
	}

	// === R:R enforcement: TP must be >= 0.7x SL distance ===
	slDist := math.Abs(fillPrice - slPrice)
	tpDistCheck := math.Abs(tpPrice - fillPrice)
	if slDist > 0 && tpDistCheck < slDist*0.7 {
		if direction == "LONG" {
			tpPrice = roundToTick(fillPrice+slDist*0.7+tickSize, tickSize)
		} else {
			tpPrice = roundToTick(fillPrice-slDist*0.7-tickSize, tickSize)
		}
	}

	// Re-verify after rounding
	tpDistFinal := math.Abs(tpPrice - fillPrice)
	if slDist > 0 && tpDistFinal < slDist*0.6 {
		if direction == "LONG" {
			tpPrice = roundToTick(fillPrice+slDist*0.8, tickSize)
		} else {
			tpPrice = roundToTick(fillPrice-slDist*0.8, tickSize)
		}
	}

	// Max TP cap: 3% from entry
	const maxTPPct = 0.03
	tpDist := math.Abs(tpPrice-fillPrice) / fillPrice
	if tpDist > maxTPPct {
		if direction == "LONG" {
			tpPrice = roundToTick(fillPrice*(1.0+maxTPPct), tickSize)
		} else {
			tpPrice = roundToTick(fillPrice*(1.0-maxTPPct), tickSize)
		}
	}

	// Final validation
	if direction == "LONG" && tpPrice <= fillPrice {
		tpPrice = roundToTick(fillPrice*(1.0+opts.MinTPPct), tickSize)
	}
	if direction == "SHORT" && tpPrice >= fillPrice {
		tpPrice = roundToTick(fillPrice*(1.0-opts.MinTPPct), tickSize)
	}

	tpQty := bybit.NormalizeQty(totalQty, qtyStep, minQty)
	grid := ExitGrid{
		StopLoss:    ExitLevel{Price: slPrice, Qty: tpQty, Kind: "stop_loss"},
		SmartSL:     smartSL,
		TakeProfits: []ExitLevel{
			{Price: tpPrice, Qty: tpQty, Kind: tpKind},
		},
	}
	return grid
}

// TPSum returns total qty allocated to take-profit levels.
func TPSum(grid ExitGrid) float64 {
	var s float64
	for _, tp := range grid.TakeProfits {
		s += tp.Qty
	}
	return s
}
