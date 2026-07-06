package risk

import "math"

// FindNearestExtrema finds the nearest local minimum (for LONG) or maximum (for SHORT)
// within the last `lookback` prices.
// For LONG: returns the lowest low (support level) as the structural SL anchor.
// For SHORT: returns the highest high (resistance level) as the structural SL anchor.
func FindNearestExtrema(prices []float64, direction string, lookback int) float64 {
	if len(prices) == 0 {
		return 0
	}

	start := 0
	if lookback > 0 && lookback < len(prices) {
		start = len(prices) - lookback
	}
	window := prices[start:]

	switch direction {
	case "LONG":
		// Find the lowest low in the window
		minPrice := math.MaxFloat64
		for _, p := range window {
			if p > 0 && p < minPrice {
				minPrice = p
			}
		}
		if minPrice == math.MaxFloat64 {
			return 0
		}
		return minPrice

	case "SHORT":
		// Find the highest high in the window
		maxPrice := 0.0
		for _, p := range window {
			if p > maxPrice {
				maxPrice = p
			}
		}
		return maxPrice

	default:
		return 0
	}
}

// FindLocalExtrema scans prices for local min/max peaks.
// Returns (localMin, localMax) where localMin is the lowest valley and localMax is the highest peak.
func FindLocalExtrema(prices []float64) (float64, float64) {
	if len(prices) < 3 {
		return 0, 0
	}

	minPrice := math.MaxFloat64
	maxPrice := 0.0

	for _, p := range prices {
		if p > 0 {
			if p < minPrice {
				minPrice = p
			}
			if p > maxPrice {
				maxPrice = p
			}
		}
	}

	if minPrice == math.MaxFloat64 {
		minPrice = 0
	}
	return minPrice, maxPrice
}
