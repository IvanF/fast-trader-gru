package liquidity

import (
	"github.com/fast-trader-gru/oms_execution/internal/models"
)

// OrderbookMomentum tracks buy/sell pressure changes over time per symbol.
// Unlike static OBI (single snapshot), momentum captures the RATE OF CHANGE
// of orderbook imbalance — detecting when pressure is building or fading.
//
// Usage:
//   1. Call Update(symbol, ob) on every orderbook tick
//   2. Call Momentum(symbol) to get current pressure velocity
//   3. Call PressureShift(symbol) to detect regime changes
//
// Zero-allocation on hot path: reuses pre-allocated ring buffers.
type OrderbookMomentum struct {
	// Per-symbol ring buffer of OBI snapshots (oldest → newest)
	obHistory map[string][]float64 // symbol → rolling OBI values (last N snapshots)
	obIdx     map[string]int       // symbol → current write index
	maxLen    int                   // ring buffer capacity
}

// NewOrderbookMomentum creates a tracker with given ring buffer size.
// Recommended: 20 snapshots (covers ~10s at 500ms update interval).
func NewOrderbookMomentum(maxLen int) *OrderbookMomentum {
	if maxLen <= 0 {
		maxLen = 20
	}
	return &OrderbookMomentum{
		obHistory: make(map[string][]float64, 64),
		obIdx:     make(map[string]int, 64),
		maxLen:    maxLen,
	}
}

// Update records current OBI from orderbook snapshot.
// Call on every orderbook update for each tracked symbol.
func (om *OrderbookMomentum) Update(symbol string, ob models.OrderbookSnapshot) {
	obi := ComputeOBI(ob)

	history := om.obHistory[symbol]
	if history == nil {
		history = make([]float64, om.maxLen)
		om.obHistory[symbol] = history
		om.obIdx[symbol] = 0
	}

	idx := om.obIdx[symbol]
	history[idx%om.maxLen] = obi
	om.obIdx[symbol] = idx + 1
}

// ComputeOBI calculates OrderBook Imbalance from orderbook snapshot.
// OBI ∈ [-1, 1]: +1 = all bids, -1 = all asks, 0 = balanced.
// Uses top-5 levels for depth-weighted pressure.
func ComputeOBI(ob models.OrderbookSnapshot) float64 {
	bidV := 0.0
	depth := min(5, len(ob.Bids))
	for i := 0; i < depth; i++ {
		s := ParseSize(ob.Bids[i].Size)
		bidV += s
	}

	askV := 0.0
	depth = min(5, len(ob.Asks))
	for i := 0; i < depth; i++ {
		s := ParseSize(ob.Asks[i].Size)
		askV += s
	}

	total := bidV + askV
	if total <= 0 {
		return 0
	}
	return (bidV - askV) / total
}

// Momentum returns the velocity of OBI change (delta OBI / time).
// Positive = buying pressure increasing.
// Negative = selling pressure increasing.
// Returns 0 if insufficient history.
func (om *OrderbookMomentum) Momentum(symbol string) float64 {
	history := om.obHistory[symbol]
	if history == nil {
		return 0
	}

	count := 0
	if om.obIdx[symbol] > om.maxLen {
		count = om.maxLen
	} else {
		count = om.obIdx[symbol]
	}
	if count < 3 {
		return 0
	}

	// Simple momentum: (latest OBI - OBI from N/2 steps ago) / (N/2)
	half := count / 2
	idx := om.obIdx[symbol]
	latest := history[(idx-1)%om.maxLen]
	older := history[(idx-1-half)%om.maxLen]
	return (latest - older) / float64(half)
}

// PressureShift detects significant change in orderbook pressure regime.
// Returns:
//   - +1: strong buying pressure shift (bids absorbing asks)
//   - -1: strong selling pressure shift (asks absorbing bids)
//   -  0: stable / insufficient data
//
// Threshold: |momentum| > 0.3 (OBI changed by 0.3 in half-window)
func (om *OrderbookMomentum) PressureShift(symbol string, threshold float64) int {
	m := om.Momentum(symbol)
	if threshold <= 0 {
		threshold = 0.3
	}
	if m > threshold {
		return +1 // buying pressure increasing
	}
	if m < -threshold {
		return -1 // selling pressure increasing
	}
	return 0
}

// CurrentOBI returns the most recent OBI value for the symbol.
func (om *OrderbookMomentum) CurrentOBI(symbol string) float64 {
	history := om.obHistory[symbol]
	if history == nil || om.obIdx[symbol] == 0 {
		return 0
	}
	idx := om.obIdx[symbol]
	return history[(idx-1)%om.maxLen]
}

// ParseSize safely converts a string size to float64.
// Returns 0 on error — avoids allocation on hot path.
func ParseSize(s string) float64 {
	var v float64
	for i := 0; i < len(s); i++ {
		c := s[i]
		if c == '.' {
			// simple float parse
			var result float64
			frac := 0.1
			for j := i + 1; j < len(s); j++ {
				d := s[j]
				if d < '0' || d > '9' {
					break
				}
				result += float64(d-'0') * frac
				frac *= 0.1
			}
			// parse integer part
			for j := 0; j < i; j++ {
				d := s[j]
				if d < '0' || d > '9' {
					return 0
				}
				result = result*10 + float64(d-'0')
			}
			return result
		}
	}
	// No decimal point — integer
	for i := 0; i < len(s); i++ {
		d := s[i]
		if d < '0' || d > '9' {
			return v
		}
		v = v*10 + float64(d-'0')
	}
	_ = v
	return 0
}
