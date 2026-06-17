package bybit

import (
	"math"
	"strconv"
	"strings"
)

type LotFilters struct {
	QtyStep     float64
	MinOrderQty float64
	MaxOrderQty float64
}

type InstrumentInfo struct {
	TickSize float64
	Lot      LotFilters
}

// NormalizeQty rounds down to qty step and bumps to exchange minimum when needed.
func NormalizeQty(qty, step, minQty float64) float64 {
	if step <= 0 {
		step = 0.001
	}
	if minQty <= 0 {
		minQty = step
	}
	if qty <= 0 {
		return minQty
	}
	steps := math.Floor(qty/step + 1e-9)
	normalized := steps * step
	if normalized < minQty {
		normalized = minQty
	}
	return normalized
}

func FormatQty(qty, step float64) string {
	return strconv.FormatFloat(qty, 'f', qtyDecimals(step), 64)
}

func qtyDecimals(step float64) int {
	if step <= 0 {
		return 3
	}
	s := strconv.FormatFloat(step, 'f', -1, 64)
	if idx := strings.IndexByte(s, '.'); idx >= 0 {
		return len(s) - idx - 1
	}
	return 0
}
