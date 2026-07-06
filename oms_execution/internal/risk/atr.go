package risk

import "math"

// CalculateATR computes Average True Range over a window of mid prices.
// Uses the standard Wilder's smoothing method.
// prices: array of mid prices (bid+ask)/2, ordered oldest first.
// period: ATR lookback (default 14).
func CalculateATR(prices []float64, period int) float64 {
	if len(prices) < 2 || period < 1 {
		return 0
	}

	// Build True Range array from consecutive price changes
	trs := make([]float64, 0, len(prices)-1)
	for i := 1; i < len(prices); i++ {
		change := math.Abs(prices[i] - prices[i-1])
		trs = append(trs, change)
	}

	if len(trs) < period {
		period = len(trs)
	}
	if period == 0 {
		return 0
	}

	// Wilder's smoothing: initial ATR = SMA of first `period` TRs
	sum := 0.0
	for i := 0; i < period; i++ {
		sum += trs[i]
	}
	atr := sum / float64(period)

	// Smooth subsequent values
	for i := period; i < len(trs); i++ {
		atr = (atr*float64(period-1) + trs[i]) / float64(period)
	}

	return atr
}

// CalculateATRFromPositionHistory computes ATR from the price history stored in a position.
// prices: array of mid prices captured at regular intervals.
func CalculateATRFromPositionHistory(prices []float64) float64 {
	return CalculateATR(prices, 14)
}

// CalculateATROrderbook computes ATR from orderbook mid prices over the last N snapshots.
func CalculateATROrderbook(mids []float64) float64 {
	return CalculateATR(mids, 14)
}

// EstimateATRFromSpread estimates a rough ATR from the current spread.
// This is a fallback when we don't have historical price data.
// ATR ≈ spread * 5 (conservative estimate for 1-minute volatility).
func EstimateATRFromSpread(bid, ask float64) float64 {
	spread := ask - bid
	if spread <= 0 || bid <= 0 {
		return 0
	}
	return spread * 5
}
