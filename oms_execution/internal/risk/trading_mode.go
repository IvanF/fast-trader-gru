package risk

import "github.com/fast-trader-gru/oms_execution/internal/models"

// TradingMode defines the execution strategy for a position.
type TradingMode int

const (
	NormalMode        TradingMode = 0
	HFTScalpingMode   TradingMode = 1
)

// DetectTradingMode evaluates current market microstructure and returns the optimal
// execution strategy for the given instrument. No hardcoded ticker names.
// It uses three factors:
//   - Normalized ATR: if the instrument moves >threshold% per candle
//   - Orderbook spread: if spread >threshold%, market is too thin for normal execution
//   - Orderbook momentum: if |momentum| > 0.3, pressure is shifting fast (HFT territory)
//
// Zero-allocation design: only reads existing fields, no heap allocations.
func DetectTradingMode(atrPct float64, spreadPct float64, momentum float64, volThreshold float64, spreadThreshold float64) TradingMode {
	if atrPct > volThreshold || spreadPct > spreadThreshold || momentum > 0.3 || momentum < -0.3 {
		return HFTScalpingMode
	}
	return NormalMode
}

// ModeTimeStopSec returns the appropriate time-stop for the given trading mode.
func ModeTimeStopSec(mode TradingMode, normalSec, hftSec int) int {
	if mode == HFTScalpingMode {
		return hftSec
	}
	return normalSec
}

// ModeBreakevenSec returns the appropriate breakeven trigger time for the given mode.
func ModeBreakevenSec(mode TradingMode, normalSec, hftSec int) int {
	if mode == HFTScalpingMode {
		return hftSec
	}
	return normalSec
}

// ModeName returns a human-readable name for the trading mode.
func ModeName(mode TradingMode) string {
	switch mode {
	case HFTScalpingMode:
		return "HFT_SCALPING"
	default:
		return "NORMAL"
	}
}

// SignalHasVolatility returns the volatility multiplier from a signal.
// Falls back to 1.0 if not set.
func SignalVolatilityMultiplier(sig models.TradeSignal) float64 {
	if sig.VolatilityMultiplier > 0 {
		return sig.VolatilityMultiplier
	}
	return 1.0
}
