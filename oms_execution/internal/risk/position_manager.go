package risk

import (
	"fmt"
)

const (
	TimeStopCandles   = 4     // Candles held before time-stop eligible
	BreakevenFeeBuffer = 0.0015 // 0.15% buffer for breakeven SL
	ChandelierATRMult = 2.5   // ATR multiplier for chandelier exit
	ScaleOutR         = 1.0   // R-level for scale-out trigger
	BreakevenR        = 1.5   // R-level for breakeven trigger
	ChandelierR       = 2.0   // R-level for chandelier activation
	ScaleOutPct       = 0.50  // 50% of position to scale out (was 25%)
)

// TradeState tracks the state of an open position for the PositionManager.
type TradeState struct {
	EntryPrice     float64
	CurrentPrice   float64
	SlPrice        float64
	OriginalRisk   float64   // Initial SL distance in price
	Direction      string
	EntryCandleIdx int       // Candle index at entry
	Size           float64
	InitialSize    float64
	ScaledOut      bool      // Has scale-out been executed?
	BreakevenSet   bool      // Has breakeven SL been set?
	CandleHigh     float64   // Current candle high (for chandelier)
	CandleLow      float64   // Current candle low
	PriceHistory   []float64 // Rolling prices for ATR
}

// PositionTrigger enumerates the possible trigger types.
type PositionTrigger int

const (
	TriggerNone PositionTrigger = iota
	TriggerTimeStop
	TriggerScaleOut
	TriggerBreakeven
	TriggerChandelierExit
)

// PositionAction describes what action to take.
type PositionAction struct {
	Type     PositionTrigger
	Action   string   // "close_full", "close_partial", "move_sl"
	Reason   string
	SlPrice  float64  // For move_sl actions
	ClosePct float64  // For close_partial actions (0.0-1.0)
}

// CurrentR calculates the current R-multiple using ORIGINAL risk (not the mutating SL).
// currentR = unrealized_pnl / original_risk
func CurrentR(entryPrice, currentPrice, originalRisk float64, direction string) float64 {
	if originalRisk <= 0 {
		return 0
	}

	var unrealizedPnl float64
	switch direction {
	case "LONG":
		unrealizedPnl = currentPrice - entryPrice
	case "SHORT":
		unrealizedPnl = entryPrice - currentPrice
	}

	return unrealizedPnl / originalRisk
}

// ManageOpenTrade checks all 4 triggers and returns the first matching action.
// This should be called on each new candle or tick.
func ManageOpenTrade(
	trade *TradeState,
	currentPrice float64,
	candleHigh, candleLow float64,
	candleIdx int,
	currentVolume float64,
	smaVolume float64,
) *PositionAction {

	trade.CurrentPrice = currentPrice
	trade.CandleHigh = candleHigh
	trade.CandleLow = candleLow

	currentR := CurrentR(trade.EntryPrice, currentPrice, trade.OriginalRisk, trade.Direction)
	candlesHeld := candleIdx - trade.EntryCandleIdx

	// ==========================================
	// TRIGGER 1: Time-Stop
	// If held >= 4 candles AND current_R < 0.5 AND no volume spike
	// ==========================================
	if candlesHeld >= TimeStopCandles && currentR < 0.5 {
		volumeSpike := currentVolume > smaVolume*1.5
		if !volumeSpike {
			return &PositionAction{
				Type:     TriggerTimeStop,
				Action:   "close_full",
				Reason:   fmt.Sprintf("Time-Stop: %d candles, R=%.2f, no volume spike", candlesHeld, currentR),
			}
		}
	}

	// ==========================================
	// TRIGGER 2: Scale Out at 1.0R
	// Close 25% of position when price reaches 1R for the first time
	// ==========================================
	if currentR >= ScaleOutR && !trade.ScaledOut {
		return &PositionAction{
			Type:      TriggerScaleOut,
			Action:    "close_partial",
			Reason:    fmt.Sprintf("Scale-Out at 1.0R (current_R=%.2f)", currentR),
			ClosePct:  ScaleOutPct,
		}
	}

	// ==========================================
	// TRIGGER 3: Breakeven at 1.5R
	// Move SL to entry ± 0.15% (fee buffer)
	// ==========================================
	if currentR >= BreakevenR && !trade.BreakevenSet {
		var newSL float64
		switch trade.Direction {
		case "LONG":
			newSL = trade.EntryPrice * (1 + BreakevenFeeBuffer)
		case "SHORT":
			newSL = trade.EntryPrice * (1 - BreakevenFeeBuffer)
		}
		return &PositionAction{
			Type:   TriggerBreakeven,
			Action: "move_sl",
			Reason: fmt.Sprintf("Breakeven at 1.5R (current_R=%.2f)", currentR),
			SlPrice: newSL,
		}
	}

	// ==========================================
	// TRIGGER 4: Chandelier Exit at >= 2.0R
	// Dynamic trailing: High - 2.5*ATR (LONG) / Low + 2.5*ATR (SHORT)
	// Only at candle close (when candleHigh/candleLow are updated)
	// ==========================================
	if currentR >= ChandelierR {
		atr := CalculateATR(trade.PriceHistory, 14)
		if atr <= 0 {
			// Fallback: estimate from spread
			if trade.CurrentPrice > 0 {
				atr = trade.CurrentPrice * 0.001 // Conservative 0.1%
			}
		}
		if atr > 0 {
			var chandelierSL float64
			switch trade.Direction {
			case "LONG":
				chandelierSL = candleHigh - atr*ChandelierATRMult
			case "SHORT":
				chandelierSL = candleLow + atr*ChandelierATRMult
			}

			// Only move SL tighter, never wider
			if trade.Direction == "LONG" && chandelierSL > trade.SlPrice {
				return &PositionAction{
					Type:   TriggerChandelierExit,
					Action: "move_sl",
					Reason: fmt.Sprintf("Chandelier at %.1fR (current_R=%.2f, ATR=%.4f)", currentR, currentR, atr),
					SlPrice: chandelierSL,
				}
			}
			if trade.Direction == "SHORT" && chandelierSL < trade.SlPrice {
				return &PositionAction{
					Type:   TriggerChandelierExit,
					Action: "move_sl",
					Reason: fmt.Sprintf("Chandelier at %.1fR (current_R=%.2f, ATR=%.4f)", currentR, currentR, atr),
					SlPrice: chandelierSL,
				}
			}
		}
	}

	return nil
}
