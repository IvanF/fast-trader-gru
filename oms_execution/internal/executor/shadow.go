package executor

import (
	"context"
	"math"
	"sync"
	"time"

	"github.com/fast-trader-gru/oms_execution/internal/models"
)

type ShadowPosition struct {
	Symbol     string
	Direction  string
	EntryPrice float64
	EntryTime  int64
	StopLoss   float64
	TakeProfit float64
	Qty        float64
	Signal     models.TradeSignal
}

type ShadowEngine struct {
	mu        sync.RWMutex
	positions map[string]*ShadowPosition
	lastOpen  map[string]int64
	logger    interface{ Info(string, ...interface{}) }
	resultsCh string
}

const shadowCooldownMs = 60000

func NewShadowEngine(logger interface{ Info(string, ...interface{}) }, resultsCh string) *ShadowEngine {
	return &ShadowEngine{
		positions: make(map[string]*ShadowPosition),
		lastOpen:  make(map[string]int64),
		logger:    logger,
		resultsCh: resultsCh,
	}
}

func (se *ShadowEngine) OpenPosition(signal models.TradeSignal, fillPrice float64, slPrice, tpPrice, qty float64) bool {
	se.mu.Lock()
	defer se.mu.Unlock()

	now := time.Now().UnixMilli()
	if last, ok := se.lastOpen[signal.Symbol]; ok && now-last < shadowCooldownMs {
		return false
	}
	if _, exists := se.positions[signal.Symbol]; exists {
		return false
	}

	se.positions[signal.Symbol] = &ShadowPosition{
		Symbol:     signal.Symbol,
		Direction:  signal.Direction,
		EntryPrice: fillPrice,
		EntryTime:  now,
		StopLoss:   slPrice,
		TakeProfit: tpPrice,
		Qty:        qty,
		Signal:     signal,
	}
	se.lastOpen[signal.Symbol] = now
	se.logger.Info("SHADOW: position opened",
		"symbol", signal.Symbol, "direction", signal.Direction,
		"entry", fillPrice, "sl", slPrice, "tp", tpPrice,
	)
	return true
}

func (se *ShadowEngine) ProcessPriceUpdate(ctx context.Context, symbol string, price float64, publishFn func(ctx context.Context, channel string, msg interface{})) {
	se.mu.Lock()
	pos, ok := se.positions[symbol]
	if !ok || price <= 0 {
		se.mu.Unlock()
		return
	}

	var closePrice float64
	var closeReason string
	closed := false

	if pos.Direction == "LONG" {
		if price <= pos.StopLoss {
			closePrice = pos.StopLoss * 1.0002
			closeReason = "stop_loss"
			closed = true
		} else if price >= pos.TakeProfit {
			closePrice = pos.TakeProfit
			closeReason = "take_profit"
			closed = true
		}
	} else {
		if price >= pos.StopLoss {
			closePrice = pos.StopLoss * 0.9998
			closeReason = "stop_loss"
			closed = true
		} else if price <= pos.TakeProfit {
			closePrice = pos.TakeProfit
			closeReason = "take_profit"
			closed = true
		}
	}

	elapsedMs := time.Now().UnixMilli() - pos.EntryTime
	if !closed && elapsedMs > 30*60*1000 {
		closePrice = price
		closeReason = "shadow_time_stop"
		closed = true
	}

	if closed {
		delete(se.positions, symbol)
		se.mu.Unlock()
		pnl := calcShadowPnL(pos, closePrice)

		se.logger.Info("SHADOW: position closed",
			"symbol", symbol, "direction", pos.Direction,
			"entry", pos.EntryPrice, "exit", closePrice,
			"pnl", pnl, "reason", closeReason, "hold_ms", elapsedMs,
		)

		result := models.ExecutionResult{
			SignalID:      pos.Signal.SignalID,
			Symbol:        symbol,
			Direction:     pos.Direction,
			StateVector:   pos.Signal.StateVector,
			EntryPrice:    pos.EntryPrice,
			ExitPrice:     closePrice,
			NetPnL:        pnl,
			HoldingTimeMs: elapsedMs,
			Regime:        pos.Signal.Regime,
			ClosedAt:      time.Now().UnixMilli(),
			CloseReason:   "shadow_" + closeReason,
			ExchangePnL:   false,
		}

		publishFn(ctx, se.resultsCh, result)
		return
	}

	se.mu.Unlock()
}

func calcShadowPnL(pos *ShadowPosition, exitPrice float64) float64 {
	notional := pos.Qty * pos.EntryPrice
	entryFee := notional * 0.00055
	exitFee := pos.Qty * exitPrice * 0.00055

	// Realistic spread simulation (0.03% average for mid-cap alts)
	spreadCost := notional * 0.0003

	// Slippage simulation for market orders (0.02% average)
	slippageCost := notional * 0.0002

	var pnl float64
	if pos.Direction == "LONG" {
		pnl = (exitPrice-pos.EntryPrice)*pos.Qty - entryFee - exitFee - spreadCost - slippageCost
	} else {
		pnl = (pos.EntryPrice-exitPrice)*pos.Qty - entryFee - exitFee - spreadCost - slippageCost
	}
	return math.Round(pnl*10000) / 10000
}

func (se *ShadowEngine) ActiveCount() int {
	se.mu.RLock()
	defer se.mu.RUnlock()
	return len(se.positions)
}

func (se *ShadowEngine) Symbols() []string {
	se.mu.RLock()
	defer se.mu.RUnlock()
	symbols := make([]string, 0, len(se.positions))
	for sym := range se.positions {
		symbols = append(symbols, sym)
	}
	return symbols
}
