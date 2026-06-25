package grid

import (
	"math"
)

// CalculateExitPrice computes the exact exit price that covers entry fee, exit fee,
// and guarantees the target net profit.
//
// LONG:  P_exit = P_entry × (1 + C_entry + T) / (1 - C_exit)
// SHORT: P_exit = P_entry × (1 - C_entry - T) / (1 + C_exit)
//
// The result is always rounded in the trader's favour:
//   - LONG  → ceil (take one tick more)
//   - SHORT → floor (take one tick less)
func CalculateExitPrice(
	entryPrice float64,
	direction string,
	entryFeeRate float64,
	exitFeeRate float64,
	targetNetProfitRate float64,
	tickSize float64,
) float64 {
	if entryPrice <= 0 || tickSize <= 0 {
		return 0
	}

	var raw float64
	if direction == "LONG" {
		raw = entryPrice * (1 + entryFeeRate + targetNetProfitRate) / (1 - exitFeeRate)
		return roundUp(raw, tickSize)
	}
	raw = entryPrice * (1 - entryFeeRate - targetNetProfitRate) / (1 + exitFeeRate)
	return roundDown(raw, tickSize)
}

// roundUp rounds price UP to the nearest tick (ceiling).
func roundUp(price, tickSize float64) float64 {
	if tickSize <= 0 {
		return price
	}
	return math.Ceil(price/tickSize) * tickSize
}

// roundDown rounds price DOWN to the nearest tick (floor).
func roundDown(price, tickSize float64) float64 {
	if tickSize <= 0 {
		return price
	}
	return math.Floor(price/tickSize) * tickSize
}
