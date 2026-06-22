package grid

import (
	"math"
	"sort"
	"strconv"

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
}

var tpLevelWeights = []float64{0.25, 0.35, 0.25, 0.15}

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
func computeSmartSL(
	direction string,
	fillPrice float64,
	ob models.OrderbookSnapshot,
	signal models.TradeSignal,
	tickSize, risk, maxSLPct float64,
) float64 {
	obi := 0.0
	if len(ob.Bids) > 0 && len(ob.Asks) > 0 {
		var bidVol, askVol float64
		for i := 0; i < 10 && i < len(ob.Bids); i++ {
			s, _ := strconv.ParseFloat(ob.Bids[i].Size, 64)
			bidVol += s
		}
		for i := 0; i < 10 && i < len(ob.Asks); i++ {
			s, _ := strconv.ParseFloat(ob.Asks[i].Size, 64)
			askVol += s
		}
		total := bidVol + askVol
		if total > 0 {
			obi = (bidVol - askVol) / total
		}
	}

	_, knifeTighten := liquidity.DetectKnife(
		signal.MacroTrend5m, signal.MacroTrend15m, obi, signal.Regime, direction,
	)

	var sl float64
	if direction == "LONG" {
		support := liquidity.FindNearestSupport(ob, fillPrice)
		if support.Price > 0 {
			sl = support.Price - tickSize*2
		} else {
			sl = fillPrice - risk
		}

		if signal.MacroTrend15m < -0.005 {
			trendTighten := math.Max(0.6, 1.0+signal.MacroTrend15m*10)
			sl = fillPrice - (fillPrice-sl)*trendTighten
		}

		sl = fillPrice - (fillPrice-sl)*knifeTighten

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
		} else {
			sl = fillPrice + risk
		}

		if signal.MacroTrend15m > 0.005 {
			trendTighten := math.Max(0.6, 1.0-signal.MacroTrend15m*10)
			sl = fillPrice + (sl-fillPrice)*trendTighten
		}

		sl = fillPrice + (sl-fillPrice)*knifeTighten

		maxSLDist := fillPrice * maxSLPct
		maxSL := fillPrice + maxSLDist
		if sl > maxSL {
			sl = maxSL
		}

		if sl <= fillPrice {
			sl = fillPrice + risk
		}
	}

	return roundToTick(sl, tickSize)
}

// BuildExitGrid places SL/TP relative to actual fill, using smart S/R-aware SL + liquidity walls for TP.
func BuildExitGrid(
	direction string,
	fillPrice, plannedEntry, plannedSL float64,
	ob models.OrderbookSnapshot,
	signal models.TradeSignal,
	tickSize, totalQty, qtyStep, minQty float64,
	opts ExitGridOptions,
) ExitGrid {
	tpBudgetPct := opts.TPBudgetPct
	vm := signal.VolatilityMultiplier
	if vm <= 0 {
		vm = 1.0
	}

	minRisk := fillPrice * opts.MinSLPct
	baseRisk := baseGridSpacing * vm * fillPrice
	if baseRisk < minRisk {
		baseRisk = minRisk
	}

	slPrice := computeSmartSL(
		direction, fillPrice, ob, signal, tickSize, baseRisk, opts.MaxSLPct,
	)

	risk := math.Abs(fillPrice - slPrice)
	if risk < minRisk {
		risk = minRisk
	}

	profitDist := 0.0
	if direction == "LONG" && fillPrice > plannedEntry {
		profitDist = fillPrice - plannedEntry
	} else if direction == "SHORT" && fillPrice < plannedEntry {
		profitDist = plannedEntry - fillPrice
	}
	if profitDist > 0 && risk > minRisk {
		tpRisk := risk - profitDist*0.5
		if tpRisk < minRisk {
			tpRisk = minRisk
		}
		risk = tpRisk
	}

	timeStopSec := opts.TimeStopSec
	if timeStopSec <= 0 {
		timeStopSec = 3600
	}
	maxTPDist := MaxTPDistance(fillPrice, vm, timeStopSec, signal.Regime)

	rangePct := baseGridSpacing * vm * 4
	wallOffset := tickSize * 2

	grid := ExitGrid{
		StopLoss: ExitLevel{Price: slPrice, Kind: "stop_loss"},
	}

	// Hybrid ladder: near heuristic levels first, then ML formula / liquidity extensions.
	grid.TakeProfits = buildHeuristicTPs(direction, fillPrice, ob, signal, risk, rangePct, wallOffset, totalQty, qtyStep, minQty, tickSize, vm, tpBudgetPct, maxTPDist)
	if len(signal.TakeProfits) > 0 {
		grid.TakeProfits = append(grid.TakeProfits, tpsFromSignal(signal.TakeProfits, direction, fillPrice, plannedEntry, totalQty, qtyStep, minQty, tickSize, tpBudgetPct)...)
	}
	if len(signal.TPPrices) > 0 {
		grid.TakeProfits = append(grid.TakeProfits, tpsFromLiquidityPrices(signal.TPPrices, direction, fillPrice, plannedEntry, totalQty, qtyStep, minQty, tickSize, tpBudgetPct)...)
	}

	grid.TakeProfits = dedupeAndAllocateTPs(direction, fillPrice, grid.TakeProfits, totalQty, qtyStep, minQty, tickSize)
	grid.TakeProfits = applyTPPriceFloors(direction, fillPrice, grid.TakeProfits, opts.MinTPPct, opts.FeeBreakevenPct, tickSize)
	grid.TakeProfits = FilterTPLevelsByMaxDistance(direction, fillPrice, grid.TakeProfits, opts.MaxTPPct, tickSize)
	if len(grid.TakeProfits) == 0 {
		grid.TakeProfits = buildHeuristicTPs(direction, fillPrice, ob, signal, risk, rangePct, wallOffset, totalQty, qtyStep, minQty, tickSize, vm, tpBudgetPct, maxTPDist)
		grid.TakeProfits = applyTPPriceFloors(direction, fillPrice, grid.TakeProfits, opts.MinTPPct, opts.FeeBreakevenPct, tickSize)
	}
	if tpBudgetPct <= 0 {
		tpBudgetPct = defaultTPBudgetPct
	}
	if tpBudgetPct > 0.65 {
		tpBudgetPct = 0.65
	}
	finalizeGridAllocation(&grid, totalQty, qtyStep, minQty, tpBudgetPct)
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
		return nil
	}

	var out []ExitLevel

	beQty := qtyFromWeight(tpBudget, tpLevelWeights[0], qtyStep, minQty)
	if beQty > 0 {
		out = append(out, ExitLevel{Price: fillPrice, Qty: beQty, Kind: "breakeven"})
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

	rMult := 2.0
	switch signal.Regime {
	case "Trending":
		rMult = 2.5
	case "Breakout":
		rMult = 3.0
	case "Choppy":
		rMult = 1.5
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
			if capPrice > 0 && capPrice > rPrice {
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
	if maxTPQty < minQty {
		grid.TakeProfits = nil
		grid.StopLoss.Qty = totalQty
		return
	}

	var tpSum float64
	for i := range grid.TakeProfits {
		grid.TakeProfits[i].Qty = bybit.NormalizeQty(grid.TakeProfits[i].Qty, qtyStep, minQty)
		tpSum += grid.TakeProfits[i].Qty
	}

	if tpSum > maxTPQty && tpSum > 0 {
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
