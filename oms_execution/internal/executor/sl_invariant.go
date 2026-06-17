package executor

import "math"

// slWouldWiden reports whether replacing currentSL with newSL increases risk distance from entry.
func slWouldWiden(direction string, currentSL, newSL float64) bool {
	if currentSL <= 0 || newSL <= 0 {
		return false
	}
	if direction == "LONG" {
		return newSL < currentSL
	}
	if direction == "SHORT" {
		return newSL > currentSL
	}
	return false
}

// clampSLTightenOnly returns the tighter of current and proposed SL (never widens).
func clampSLTightenOnly(direction string, currentSL, proposed float64) float64 {
	if currentSL <= 0 {
		return proposed
	}
	if direction == "LONG" {
		return math.Max(proposed, currentSL)
	}
	if direction == "SHORT" {
		return math.Min(proposed, currentSL)
	}
	return proposed
}
