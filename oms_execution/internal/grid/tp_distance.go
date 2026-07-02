package grid

import "math"

// ExitGridOptions controls TP ladder pricing floors and ceilings.
type ExitGridOptions struct {
	TPBudgetPct        float64
	MinTPPct           float64
	MaxTPPct           float64
	FeeBreakevenPct    float64
	MinSLPct           float64
	MaxSLPct           float64
	TimeStopSec        int
	EntryFeeRate       float64 // taker fee rate for entry (e.g. 0.00055)
	ExitFeeRate        float64 // maker fee rate for exit (e.g. 0.0002)
	TargetNetProfitPct float64 // target net profit as fraction (e.g. 0.002 = 0.2%)
}

// MaxTPDistance calculates the maximum realistic TP distance based on
// volatility (vm), time horizon (timeStopSec), and regime.
// Uses sqrt-time scaling: longer holds allow proportionally larger moves.
func MaxTPDistance(fillPrice, vm float64, timeStopSec int, regime string) float64 {
	if timeStopSec <= 0 {
		timeStopSec = 3600
	}
	if vm <= 0 {
		vm = 1.0
	}

	baseSpacing := 0.002
	timeFactor := math.Sqrt(float64(timeStopSec) / 3600.0)

	regimeMult := 1.0
	switch regime {
	case "Trending":
		regimeMult = 1.2
	case "Breakout":
		regimeMult = 1.5
	case "Choppy":
		regimeMult = 0.8
	}

	maxMove := fillPrice * baseSpacing * vm * timeFactor * regimeMult * 3.0
	if maxMove < fillPrice*0.003 {
		maxMove = fillPrice * 0.003
	}
	if maxMove > fillPrice*0.05 {
		maxMove = fillPrice * 0.05
	}
	return maxMove
}

// FeeAwareBreakevenPrice returns the minimum profitable exit (covers fees + micro-slippage).
func FeeAwareBreakevenPrice(fillPrice float64, direction string, feePct, tickSize float64, entryFeeRate, exitFeeRate float64) float64 {
	if fillPrice <= 0 || feePct <= 0 {
		return fillPrice
	}
	if tickSize <= 0 {
		tickSize = 0.0001
	}
	if entryFeeRate <= 0 {
		entryFeeRate = 0.00055
	}
	if exitFeeRate <= 0 {
		exitFeeRate = 0.0002
	}
	switch direction {
	case "LONG":
		be := fillPrice * (1.0 + entryFeeRate) / (1.0 - exitFeeRate)
		if be <= fillPrice {
			be = fillPrice * (1.0 + feePct)
		}
		return roundToTick(be, tickSize)
	case "SHORT":
		be := fillPrice * (1.0 - entryFeeRate) / (1.0 + exitFeeRate)
		if be >= fillPrice {
			be = fillPrice * (1.0 - feePct)
		}
		return roundToTick(be, tickSize)
	default:
		return fillPrice
	}
}

// EnforceMinTPDistance pushes TP away from fill if closer than minPct.
func EnforceMinTPDistance(fillPrice, tpPrice float64, direction string, minPct, tickSize float64) float64 {
	if fillPrice <= 0 || minPct <= 0 {
		return tpPrice
	}
	if tickSize <= 0 {
		tickSize = 0.0001
	}
	minDist := fillPrice * minPct

	switch direction {
	case "LONG":
		floor := fillPrice + minDist
		if tpPrice <= 0 || tpPrice < floor {
			return roundToTick(floor, tickSize)
		}
	case "SHORT":
		ceiling := fillPrice - minDist
		if tpPrice <= 0 || tpPrice > ceiling {
			return roundToTick(ceiling, tickSize)
		}
	}
	return roundToTick(tpPrice, tickSize)
}

// TPDistancePct returns absolute distance between fill and TP as fraction of fill.
func TPDistancePct(fillPrice, tpPrice float64) float64 {
	if fillPrice <= 0 || tpPrice <= 0 {
		return 0
	}
	return math.Abs(tpPrice-fillPrice) / fillPrice
}

// FilterTPLevelsByMaxDistance drops TP levels farther than maxPct from fill.
func FilterTPLevelsByMaxDistance(
	direction string,
	fillPrice float64,
	levels []ExitLevel,
	maxPct, tickSize float64,
) []ExitLevel {
	if maxPct <= 0 || len(levels) == 0 {
		return levels
	}
	out := make([]ExitLevel, 0, len(levels))
	for _, lv := range levels {
		if TPDistancePct(fillPrice, lv.Price) > maxPct {
			continue
		}
		lv.Price = roundToTick(lv.Price, tickSize)
		out = append(out, lv)
	}
	return out
}

func applyTPPriceFloors(
	direction string,
	fillPrice float64,
	levels []ExitLevel,
	minTPPct, feeBreakevenPct, tickSize float64,
	entryFeeRate, exitFeeRate float64,
) []ExitLevel {
	if len(levels) == 0 {
		return levels
	}
	minBE := FeeAwareBreakevenPrice(fillPrice, direction, feeBreakevenPct, tickSize, entryFeeRate, exitFeeRate)
	out := make([]ExitLevel, 0, len(levels))
	for _, lv := range levels {
		if lv.Kind == "wall" {
			lv.Price = roundToTick(lv.Price, tickSize)
			if direction == "LONG" {
				lv.Price = math.Max(lv.Price, fillPrice+tickSize)
			} else {
				lv.Price = math.Min(lv.Price, fillPrice-tickSize)
			}
		} else {
			lv.Price = EnforceMinTPDistance(fillPrice, lv.Price, direction, minTPPct, tickSize)
			if lv.Kind == "breakeven" {
				switch direction {
				case "LONG":
					if lv.Price < minBE {
						lv.Price = minBE
					}
				case "SHORT":
					if lv.Price > minBE {
						lv.Price = minBE
					}
				}
			}
		}
		out = append(out, lv)
	}

	if direction == "SHORT" {
		offset := math.Max(fillPrice*0.0005, tickSize*3)
		for i := len(out) - 1; i >= 1; i-- {
			if out[i].Price >= out[i-1].Price {
				out[i].Price = out[i-1].Price - offset
			}
		}
	} else if direction == "LONG" {
		offset := math.Max(fillPrice*0.0005, tickSize*3)
		for i := len(out) - 1; i >= 1; i-- {
			if out[i].Price <= out[i-1].Price {
				out[i].Price = out[i-1].Price + offset
			}
		}
	}
	return out
}
