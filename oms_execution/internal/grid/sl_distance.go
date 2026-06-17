package grid

import "math"

// EnforceMinSLDistance widens SL if it is closer than minPct from fill price.
func EnforceMinSLDistance(fillPrice, slPrice float64, direction string, minPct, tickSize float64) float64 {
	if fillPrice <= 0 || minPct <= 0 {
		return slPrice
	}
	minDist := fillPrice * minPct
	if tickSize <= 0 {
		tickSize = 0.0001
	}

	switch direction {
	case "LONG":
		maxSL := fillPrice - minDist
		if slPrice <= 0 || slPrice > maxSL {
			return roundToTick(maxSL, tickSize)
		}
	case "SHORT":
		minSL := fillPrice + minDist
		if slPrice <= 0 || slPrice < minSL {
			return roundToTick(minSL, tickSize)
		}
	}
	return roundToTick(slPrice, tickSize)
}

// SLDistancePct returns absolute distance between fill and SL as fraction of fill.
func SLDistancePct(fillPrice, slPrice float64) float64 {
	if fillPrice <= 0 || slPrice <= 0 {
		return 0
	}
	return math.Abs(slPrice-fillPrice) / fillPrice
}
