package grid

import (
	"math"
	"sort"

	"github.com/fast-trader-gru/oms_execution/internal/bybit"
	"github.com/fast-trader-gru/oms_execution/internal/liquidity"
	"github.com/fast-trader-gru/oms_execution/internal/models"
)

// tpBudgetPct — max share of position allocated to TP ladder; remainder goes to SL (full coverage).
const defaultTPBudgetPct = 0.35

// ExitLevel is one reduce-only limit exit order.
type ExitLevel struct {
	Price float64
	Qty   float64
	Kind  string // breakeven | wall | trend | r_multiple | ml_tp
}

// ExitGrid holds SL + TP ladder for a filled position.
type ExitGrid struct {
	StopLoss    ExitLevel
	TakeProfits []ExitLevel
	SmartSL     bool // true if SL is placed at real S/R level
}

var tpLevelWeights = []float64{0.25, 0.40, 0.25, 0.10}

// atrSLMultiplier scales SL distance by regime: trending needs room, choppy tightens.
func atrSLMultiplier(regime string) float64 {
	switch regime {
	case "Trending":
		return 1.8
	case "Breakout":
		return 2.2
	default: // Choppy
		return 1.2
	}
}

// computeSmartSL places SL at nearest S/R level, capped by max distance, tightened for knives.
// Returns (slPrice, foundSR) — foundSR=true means SL is at a real S/R level.
func computeSmartSL(
	direction string,
	fillPrice float64,
	ob models.OrderbookSnapshot,
	signal models.TradeSignal,
	tickSize, risk, maxSLPct float64,
) (float64, bool) {
	// Knife: do NOT tighten SL — entering against trend needs wider stop to avoid noise.
	// Only TP should be tightened (quick profit capture). SL stays at S/R level.

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

// BuildExitGrid places SL + single TP at fee-aware price covering commissions + target profit.
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

	// Scale SL by volatility: calm coins get tighter SL, volatile coins get wider SL
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

	// === SL computation (unchanged logic) ===
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
	minSLTicks := 5.0
	minSLFromTicks := minSLTicks * tickSize
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

	// === TP computation — unified pipeline for ALL coins ===
	maxTPDist := opts.MaxTPPct
	if maxTPDist <= 0 {
		maxTPDist = 0.015
	}

	// Priority 1: Python-computed TP from spike-detection (orderbook walls)
	if hasPythonTP {
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

	// Priority 2: Go-side nearest level from orderbook
	if tpPrice <= 0 {
		var levelTP float64
		levelBuffer := tickSize * 5
		entryFee := opts.EntryFeeRate
		if entryFee <= 0 {
			entryFee = 0.00055
		}
		exitFee := opts.ExitFeeRate
		if exitFee <= 0 {
			exitFee = 0.0002
		}
		if direction == "LONG" {
			resistance := liquidity.FindNearestResistance(ob, fillPrice)
			if resistance.Price > 0 {
				levelTP = roundToTick(resistance.Price-levelBuffer, tickSize)
				if levelTP <= fillPrice {
					levelTP = 0
				}
			}
		} else {
			support := liquidity.FindNearestSupport(ob, fillPrice)
			if support.Price > 0 {
				levelTP = roundToTick(support.Price+levelBuffer, tickSize)
				if levelTP >= fillPrice {
					levelTP = 0
				}
			}
		}
		if levelTP > 0 {
			levelDist := math.Abs(levelTP-fillPrice) / fillPrice
			feeMinDist := math.Max((entryFee+exitFee)*2, 0.01) // At least 1% from entry
			if levelDist >= feeMinDist && levelDist <= maxTPDist {
				tpPrice = levelTP
				tpKind = "liquidity_tp"
			}
		}
	}

	// Priority 3: fee-aware TP (guaranteed minimum profit)
	if tpPrice <= 0 {
		entryFee := opts.EntryFeeRate
		if entryFee <= 0 {
			entryFee = 0.00055
		}
		exitFee := opts.ExitFeeRate
		if exitFee <= 0 {
			exitFee = 0.0002
		}
		targetProfit := opts.TargetNetProfitPct
		if targetProfit <= 0 {
			targetProfit = 0.002
		}
		if tickSize <= 0 {
			tickSize = 0.0001
		}
		tpPrice = CalculateExitPrice(fillPrice, direction, entryFee, exitFee, targetProfit, tickSize)
		if tpPrice <= 0 {
			// Absolute fallback: fill ± (fees + 0.5%)
			fallbackDist := fillPrice * (entryFee + exitFee + 0.005)
			if direction == "LONG" {
				tpPrice = roundToTick(fillPrice+fallbackDist, tickSize)
			} else {
				tpPrice = roundToTick(fillPrice-fallbackDist, tickSize)
			}
		}
	}

	// Enforce MinTPPct only for non-liquidity TPs
	if tpKind != "liquidity_tp" && tpKind != "ml_tp" {
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

	// === R:R enforcement: TP must be >= 0.5x SL distance ===
	slDist := math.Abs(fillPrice - slPrice)
	tpDistCheck := math.Abs(tpPrice - fillPrice)
	if slDist > 0 && tpDistCheck < slDist*0.5 {
		// TP too close relative to SL — widen to 0.5x SL
		if direction == "LONG" {
			tpPrice = roundToTick(fillPrice+slDist*0.5, tickSize)
		} else {
			tpPrice = roundToTick(fillPrice-slDist*0.5, tickSize)
		}
	}

	// Final validation: TP must be in profit direction
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

func buildHeuristicTPs(
	direction string,
	fillPrice float64,
	ob models.OrderbookSnapshot,
	signal models.TradeSignal,
	risk, rangePct, wallOffset, totalQty, qtyStep, minQty, tickSize, vm, tpBudgetPct, maxTPDist float64,
) []ExitLevel {
	if tpBudgetPct <= 0 {
		tpBudgetPct = defaultTPBudgetPct
	}
	tpBudget := bybit.NormalizeQty(totalQty*tpBudgetPct, qtyStep, minQty)
	if tpBudget <= 0 {
		if totalQty >= minQty {
			tpBudget = minQty
		} else {
			return nil
		}
	}

	var out []ExitLevel

	// For very small positions: single wall TP at fillPrice-risk*0.3
	if totalQty <= minQty*2 {
		var tpPrice float64
		if direction == "SHORT" {
			tpPrice = roundToTick(fillPrice-risk*0.3, tickSize)
		} else {
			tpPrice = roundToTick(fillPrice+risk*0.3, tickSize)
		}
		out = append(out, ExitLevel{Price: tpPrice, Qty: minQty, Kind: "wall"})
		return out
	}

	var wallPrice float64
	if direction == "LONG" {
		wall := liquidity.FindResistanceWall(ob, fillPrice, rangePct)
		wallPrice = roundToTick(wall.Price-wallOffset, tickSize)
		if wallPrice <= fillPrice {
			wallPrice = roundToTick(fillPrice+risk, tickSize)
		}
		if wallPrice <= fillPrice+risk*0.3 {
			wallPrice = roundToTick(fillPrice+risk*0.5, tickSize)
		}
	} else {
		wall := liquidity.FindSupportWall(ob, fillPrice, rangePct)
		wallPrice = roundToTick(wall.Price+wallOffset, tickSize)
		if wallPrice <= 0 || wallPrice >= fillPrice {
			wallPrice = roundToTick(fillPrice-risk*0.5, tickSize)
		}
	}
	wallQty := qtyFromWeight(tpBudget, tpLevelWeights[1], qtyStep, minQty)
	if wallQty > 0 && wallPrice > 0 {
		out = append(out, ExitLevel{Price: wallPrice, Qty: wallQty, Kind: "wall"})
	}

	// Quick profit TP: nearest S/R level before wall (captures early profit)
	var quickPrice float64
	if direction == "LONG" {
		// Find nearest resistance below wall
		quickPrice = roundToTick(fillPrice+risk*0.4, tickSize)
	} else {
		// Find nearest support above wall
		quickPrice = roundToTick(fillPrice-risk*0.4, tickSize)
	}
	quickQty := qtyFromWeight(tpBudget, tpLevelWeights[3], qtyStep, minQty)
	if quickQty > 0 && quickPrice > 0 {
		// Only add if different from wall
		if math.Abs(quickPrice-wallPrice) > tickSize*2 {
			out = append(out, ExitLevel{Price: quickPrice, Qty: quickQty, Kind: "quick_profit"})
		}
	}

	rMult := 1.5
	switch signal.Regime {
	case "Trending":
		rMult = 2.0
	case "Breakout":
		rMult = 2.5
	case "Choppy":
		rMult = 1.2
	}
	rMult *= vm

	var rPrice float64
	if direction == "LONG" {
		rPrice = roundToTick(fillPrice+risk*rMult, tickSize)
	} else {
		rPrice = roundToTick(fillPrice-risk*rMult, tickSize)
	}

	strongWall := liquidity.FindStrongestWallWithin(ob, direction, fillPrice, maxTPDist)
	if strongWall.Price > 0 {
		wallOffset2 := tickSize * 3
		if direction == "LONG" {
			capPrice := roundToTick(strongWall.Price-wallOffset2, tickSize)
			if capPrice > 0 && capPrice < rPrice {
				rPrice = capPrice
			}
		} else {
			capPrice := roundToTick(strongWall.Price+wallOffset2, tickSize)
			if capPrice > 0 && capPrice > rPrice && capPrice < fillPrice {
				rPrice = capPrice
			}
		}
	}

	rDist := math.Abs(rPrice - fillPrice)
	if rDist > maxTPDist {
		if direction == "LONG" {
			rPrice = roundToTick(fillPrice+maxTPDist, tickSize)
		} else {
			rPrice = roundToTick(fillPrice-maxTPDist, tickSize)
		}
	}

	rQty := qtyFromWeight(tpBudget, tpLevelWeights[2], qtyStep, minQty)
	if rQty > 0 {
		out = append(out, ExitLevel{Price: rPrice, Qty: rQty, Kind: "r_multiple"})
	}

	if signal.Regime == "Trending" || signal.Regime == "Breakout" {
		ext := risk * (rMult + 1)
		var extPrice float64
		if direction == "LONG" {
			extPrice = roundToTick(fillPrice+ext, tickSize)
		} else {
			extPrice = roundToTick(fillPrice-ext, tickSize)
		}

		if strongWall.Price > 0 {
			wallOffset3 := tickSize * 2
			if direction == "LONG" {
				capPrice := roundToTick(strongWall.Price-wallOffset3, tickSize)
				if capPrice > 0 && capPrice < extPrice {
					extPrice = capPrice
				}
			} else {
				capPrice := roundToTick(strongWall.Price+wallOffset3, tickSize)
				if capPrice > 0 && capPrice > extPrice {
					extPrice = capPrice
				}
			}
		}

		extDist := math.Abs(extPrice - fillPrice)
		if extDist > maxTPDist*1.5 {
			if direction == "LONG" {
				extPrice = roundToTick(fillPrice+maxTPDist*1.5, tickSize)
			} else {
				extPrice = roundToTick(fillPrice-maxTPDist*1.5, tickSize)
			}
		}

		extQty := qtyFromWeight(tpBudget, tpLevelWeights[3], qtyStep, minQty)
		if extQty > 0 {
			out = append(out, ExitLevel{Price: extPrice, Qty: extQty, Kind: "trend"})
		}
	}
	return out
}

func qtyFromWeight(budget, weight, qtyStep, minQty float64) float64 {
	if budget <= 0 || weight <= 0 {
		return 0
	}
	return bybit.NormalizeQty(budget*weight, qtyStep, minQty)
}

func finalizeGridAllocation(grid *ExitGrid, totalQty, qtyStep, minQty, tpBudgetPct float64) {
	originalQty := totalQty
	totalQty = bybit.NormalizeQty(totalQty, qtyStep, minQty)
	if totalQty <= 0 {
		return
	}

	maxTPQty := totalQty - minQty

	// For small positions: keep at least 2 TPs with minQty each
	if len(grid.TakeProfits) >= 2 {
		for i := range grid.TakeProfits {
			grid.TakeProfits[i].Qty = bybit.NormalizeQty(grid.TakeProfits[i].Qty, qtyStep, minQty)
			if grid.TakeProfits[i].Qty <= 0 {
				grid.TakeProfits[i].Qty = minQty
			}
		}
		var tpSum float64
		for _, tp := range grid.TakeProfits {
			tpSum += tp.Qty
		}
		// If TPs exceed maxTPQty, scale them down
		if tpSum > maxTPQty && maxTPQty > 0 {
			scale := maxTPQty / tpSum
			tpSum = 0
			trimmed := make([]ExitLevel, 0, len(grid.TakeProfits))
			for _, tp := range grid.TakeProfits {
				q := bybit.NormalizeQty(tp.Qty*scale, qtyStep, minQty)
				if q <= 0 || tpSum+q > maxTPQty {
					continue
				}
				tp.Qty = q
				trimmed = append(trimmed, tp)
				tpSum += q
			}
			if len(trimmed) < 2 && len(grid.TakeProfits) > 0 {
				trimmed = append(trimmed, grid.TakeProfits[0])
				trimmed[len(trimmed)-1].Qty = minQty
				tpSum += minQty
			}
			grid.TakeProfits = trimmed
		}
		slQty := bybit.NormalizeQty(totalQty, qtyStep, minQty)
		grid.StopLoss.Qty = slQty
		return
	}

	var tpSum float64
	for i := range grid.TakeProfits {
		grid.TakeProfits[i].Qty = bybit.NormalizeQty(grid.TakeProfits[i].Qty, qtyStep, minQty)
		tpSum += grid.TakeProfits[i].Qty
	}

	if tpSum > maxTPQty && tpSum > 0 {
		if maxTPQty > 0 {
			scale := maxTPQty / tpSum
			tpSum = 0
			trimmed := make([]ExitLevel, 0, len(grid.TakeProfits))
			for _, tp := range grid.TakeProfits {
				q := bybit.NormalizeQty(tp.Qty*scale, qtyStep, minQty)
				if q <= 0 || tpSum+q > maxTPQty {
					continue
				}
				tp.Qty = q
				trimmed = append(trimmed, tp)
				tpSum += q
			}
			grid.TakeProfits = trimmed
		} else {
			grid.TakeProfits = grid.TakeProfits[:1]
			grid.TakeProfits[0].Qty = minQty
			tpSum = minQty
		}
	}

	slQty := bybit.NormalizeQty(totalQty, qtyStep, minQty)
	if slQty <= 0 && tpSum < totalQty {
		trimmed := make([]ExitLevel, 0, len(grid.TakeProfits))
		tpSum = 0
		maxTP := totalQty - minQty
		for _, tp := range grid.TakeProfits {
			if tpSum+tp.Qty > maxTP {
				tp.Qty = bybit.NormalizeQty(maxTP-tpSum, qtyStep, minQty)
			}
			if tp.Qty <= 0 {
				continue
			}
			trimmed = append(trimmed, tp)
			tpSum += tp.Qty
		}
		grid.TakeProfits = trimmed
		slQty = bybit.NormalizeQty(totalQty-tpSum, qtyStep, minQty)
	}
	if slQty <= 0 {
		slQty = originalQty
		if slQty < minQty {
			slQty = totalQty
		}
		slQty = bybit.NormalizeQty(slQty, qtyStep, minQty)
	}
	grid.StopLoss.Qty = slQty
}

func dedupeAndAllocateTPs(
	direction string,
	fillPrice float64,
	levels []ExitLevel,
	totalQty, qtyStep, minQty, tickSize float64,
) []ExitLevel {
	type key struct {
		price float64
		kind  string
	}
	seen := make(map[key]ExitLevel)
	for _, lv := range levels {
		if lv.Qty <= 0 || lv.Price <= 0 {
			continue
		}
		if lv.Kind != "breakeven" {
			if direction == "LONG" && lv.Price <= fillPrice {
				continue
			}
			if direction == "SHORT" && lv.Price >= fillPrice {
				continue
			}
		}
		k := key{price: roundToTick(lv.Price, tickSize), kind: lv.Kind}
		if existing, ok := seen[k]; ok {
			existing.Qty += lv.Qty
			seen[k] = existing
		} else {
			lv.Price = k.price
			seen[k] = lv
		}
	}
	out := make([]ExitLevel, 0, len(seen))
	for _, lv := range seen {
		out = append(out, lv)
	}
	sort.Slice(out, func(i, j int) bool {
		if direction == "LONG" {
			return out[i].Price < out[j].Price
		}
		return out[i].Price > out[j].Price
	})
	return out
}

func tpsFromSignal(
	tps []float64,
	direction string,
	fillPrice, plannedEntry, totalQty, qtyStep, minQty, tickSize, tpBudgetPct float64,
) []ExitLevel {
	delta := fillPrice - plannedEntry
	if tpBudgetPct <= 0 {
		tpBudgetPct = defaultTPBudgetPct
	}
	tpBudget := bybit.NormalizeQty(totalQty*tpBudgetPct, qtyStep, minQty)
	if tpBudget <= 0 {
		return nil
	}

	out := make([]ExitLevel, 0, len(tps))
	for i, tp := range tps {
		if tp <= 0 {
			continue
		}
		w := 0.25
		if i < len(tpLevelWeights) {
			w = tpLevelWeights[i]
		} else if len(tps) > 0 {
			w = 1.0 / float64(len(tps))
		}
		price := roundToTick(tp+delta, tickSize)
		if direction == "LONG" && price <= fillPrice {
			continue
		}
		if direction == "SHORT" && price >= fillPrice {
			continue
		}
		kind := "ml_tp"
		switch i {
		case 0:
			kind = "breakeven"
		case 1:
			kind = "wall"
		case len(tps) - 1:
			if len(tps) > 2 {
				kind = "trend"
			} else {
				kind = "r_multiple"
			}
		case 2:
			kind = "r_multiple"
		}
		qty := qtyFromWeight(tpBudget, w, qtyStep, minQty)
		if qty <= 0 {
			continue
		}
		out = append(out, ExitLevel{Price: price, Qty: qty, Kind: kind})
	}
	return out
}

func tpsFromLiquidityPrices(
	prices []float64,
	direction string,
	fillPrice, plannedEntry, totalQty, qtyStep, minQty, tickSize, tpBudgetPct float64,
) []ExitLevel {
	delta := fillPrice - plannedEntry
	if tpBudgetPct <= 0 {
		tpBudgetPct = defaultTPBudgetPct
	}
	tpBudget := bybit.NormalizeQty(totalQty*tpBudgetPct, qtyStep, minQty)
	if tpBudget <= 0 {
		return nil
	}

	out := make([]ExitLevel, 0, len(prices))
	for i, tp := range prices {
		if tp <= 0 {
			continue
		}
		w := 0.25
		if i < len(tpLevelWeights) {
			w = tpLevelWeights[i]
		} else if len(prices) > 0 {
			w = 1.0 / float64(len(prices))
		}
		price := roundToTick(tp+delta, tickSize)
		if direction == "LONG" && price <= fillPrice {
			continue
		}
		if direction == "SHORT" && price >= fillPrice {
			continue
		}
		kind := "liquidity_tp"
		qty := qtyFromWeight(tpBudget, w, qtyStep, minQty)
		if qty <= 0 {
			continue
		}
		out = append(out, ExitLevel{Price: price, Qty: qty, Kind: kind})
	}
	return out
}

// TPSum returns total qty allocated to take-profit levels.
func TPSum(grid ExitGrid) float64 {
	var s float64
	for _, tp := range grid.TakeProfits {
		s += tp.Qty
	}
	return s
}
