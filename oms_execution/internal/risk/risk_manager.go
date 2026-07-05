package risk

import (
	"fmt"
	"math"
)

const (
	KellyFraction     = 0.5   // Half-Kelly for safety
	MaxRiskPerTrade   = 0.02  // Hard limit: 2% of deposit per trade
	MaxSLPct          = 0.005 // Hard SL cap: 0.5%
	WickBufferMult    = 1.5   // Wick buffer multiplier
	ATRMult           = 2.0   // ATR multiplier for base SL
	MaxTickSizePct    = 0.001 // Max tick size as % of price: 0.1%
)

// SignalResult is the output of process_signal.
type SignalResult struct {
	Approved       bool
	RejectReason   string
	SlPrice        float64
	TpPrice        float64
	Qty            float64
	RiskPct        float64 // Fraction of deposit risked
	KellyPct       float64
	EV             float64
	RewardRiskRatio float64
	SLDistancePct  float64
}

// ProcessSignal evaluates a trading signal using ATR-based SL, EV check, and Kelly sizing.
//
// direction: "LONG" or "SHORT"
// confidence: ML model confidence (0.0 - 1.0)
// entryPrice: planned entry price
// tpPrice: planned take-profit price
// prices: rolling price history for ATR computation (oldest first)
// accountBalance: current deposit in USD
// tickSize: exchange tick size for the symbol
func ProcessSignal(
	direction string,
	confidence float64,
	entryPrice float64,
	tpPrice float64,
	prices []float64,
	accountBalance float64,
	volMultiplier float64,
	tickSize float64,
) SignalResult {
	result := SignalResult{}

	// Validate inputs
	if entryPrice <= 0 || tpPrice <= 0 || confidence <= 0 || confidence > 1 {
		result.RejectReason = fmt.Sprintf("invalid inputs: entry=%.4f tp=%.4f conf=%.4f", entryPrice, tpPrice, confidence)
		return result
	}
	if accountBalance <= 0 {
		result.RejectReason = "account balance <= 0"
		return result
	}

	// ============================================
	// TICK SIZE FILTER: reject if tick > 0.1% of price
	// ============================================
	if tickSize > 0 && entryPrice > 0 {
		tickSizePct := tickSize / entryPrice
		if tickSizePct > MaxTickSizePct {
			result.RejectReason = fmt.Sprintf("tick size too large: %.4f%% > %.2f%%", tickSizePct*100, MaxTickSizePct*100)
			return result
		}
	}

	// ============================================
	// STEP 1: Dynamic SL from ATR + Extrema + Wick
	// ============================================
	atr := CalculateATR(prices, 14)
	if atr <= 0 {
		// Fallback: estimate from price range
		if len(prices) >= 2 {
			atr = math.Abs(prices[len(prices)-1]-prices[0]) / float64(len(prices)-1) * 5
		}
		if atr <= 0 {
			result.RejectReason = "cannot compute ATR"
			return result
		}
	}

	extrema := FindNearestExtrema(prices, direction, 20)
	wickBuffer := 0.0
	if len(prices) >= 2 {
		// Estimate wick from recent price range
		recent := prices
		if len(prices) > 20 {
			recent = prices[len(prices)-20:]
		}
		minP, maxP := FindLocalExtrema(recent)
		if maxP > minP && minP > 0 {
			wickBuffer = (maxP - minP) / float64(len(recent)) * WickBufferMult
		}
	}

	// Base SL: safest of ATR-based and structure-based
	var slPrice float64
	switch direction {
	case "LONG":
		atrSL := entryPrice - atr*ATRMult
		structSL := extrema
		if structSL > 0 && structSL < atrSL {
			slPrice = structSL - wickBuffer
		} else {
			slPrice = atrSL - wickBuffer
		}
		// SL must be below entry for LONG
		if slPrice >= entryPrice {
			slPrice = entryPrice - atr*ATRMult - wickBuffer
		}
	case "SHORT":
		atrSL := entryPrice + atr*ATRMult
		structSL := extrema
		if structSL > 0 && structSL > atrSL {
			slPrice = structSL + wickBuffer
		} else {
			slPrice = atrSL + wickBuffer
		}
		// SL must be above entry for SHORT
		if slPrice <= entryPrice {
			slPrice = entryPrice + atr*ATRMult + wickBuffer
		}
	}

	if slPrice <= 0 {
		result.RejectReason = "SL price <= 0"
		return result
	}

	// ============================================
	// STEP 2: Hard Limit — SL distance ≤ 1.5%
	// ============================================
	slDistancePct := math.Abs(entryPrice-slPrice) / entryPrice
	if slDistancePct > MaxSLPct {
		result.RejectReason = fmt.Sprintf("SL too wide: %.2f%% > %.2f%% max", slDistancePct*100, MaxSLPct*100)
		result.SLDistancePct = slDistancePct
		return result
	}

	// ============================================
	// STEP 3: R:R and EV check
	// ============================================
	tpDistancePct := math.Abs(tpPrice-entryPrice) / entryPrice
	if slDistancePct <= 0 {
		result.RejectReason = "SL distance is zero"
		return result
	}

	rr := tpDistancePct / slDistancePct
	ev := (confidence * rr) - ((1 - confidence) * 1)

	result.RewardRiskRatio = rr
	result.EV = ev
	result.SLDistancePct = slDistancePct

	if ev <= 0 {
		result.RejectReason = fmt.Sprintf("negative EV: %.4f (conf=%.3f RR=%.2f)", ev, confidence, rr)
		return result
	}

	// ============================================
	// STEP 4: Kelly criterion position sizing
	// ============================================
	kellyPct := confidence - ((1 - confidence) / rr)
	adjustedKelly := kellyPct * KellyFraction

	// Spec: risk = min(K * 0.5, 0.02) — half-Kelly capped at hard 2%
	finalRiskPct := math.Min(adjustedKelly, MaxRiskPerTrade)

	// Volatility penalty: high-vol symbols (volMult > 1.5) get smaller positions
	if volMultiplier > 1.5 {
		volPenalty := 1.5 / volMultiplier // volMult=2.0 → penalty=0.75, volMult=3.0 → penalty=0.5
		finalRiskPct *= volPenalty
	}

	if finalRiskPct <= 0 {
		result.RejectReason = fmt.Sprintf("Kelly suggests no position: kelly=%.4f adj=%.4f", kellyPct, adjustedKelly)
		return result
	}

	// Position size: (Balance * Risk%) / SL_distance_pct
	qty := (accountBalance * finalRiskPct) / slDistancePct

	result.Approved = true
	result.SlPrice = slPrice
	result.TpPrice = tpPrice
	result.Qty = qty
	result.RiskPct = finalRiskPct
	result.KellyPct = kellyPct

	return result
}
