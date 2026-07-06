package risk

// CalculateAvgWick computes the average wick length from OHLC candle data.
// Wick = average of upper wick (High - max(Open, Close)) and lower wick (min(Open, Close) - Low).
// Returns the average wick size per candle over the last `period` candles.
//
// candles: each candle is [Open, High, Low, Close] (4-element arrays).
func CalculateAvgWick(candles [][]float64, period int) float64 {
	if len(candles) == 0 || period <= 0 {
		return 0
	}

	start := 0
	if period < len(candles) {
		start = len(candles) - period
	}

	totalWick := 0.0
	count := 0

	for _, c := range candles[start:] {
		if len(c) < 4 {
			continue
		}
		open, high, low, close_ := c[0], c[1], c[2], c[3]
		if high <= low || open == 0 || close_ == 0 {
			continue
		}

		upperWick := high - max2(open, close_)
		lowerWick := min2(open, close_) - low

		totalWick += upperWick + lowerWick
		count++
	}

	if count == 0 {
		return 0
	}

	// Average wick per candle (upper + lower combined)
	return totalWick / float64(count)
}

// EstimateWickFromSpread estimates wick buffer from orderbook spread.
// On thin books, the spread itself represents potential wick risk.
func EstimateWickFromSpread(bid, ask float64) float64 {
	spread := ask - bid
	if spread <= 0 || bid <= 0 {
		return 0
	}
	return spread * 0.5 // Half-spread as conservative wick estimate
}

func max2(a, b float64) float64 {
	if a > b {
		return a
	}
	return b
}

func min2(a, b float64) float64 {
	if a < b {
		return a
	}
	return b
}
