package subscriber

import (
	"context"
	"encoding/json"
	"log/slog"
	"math"
	"sync"
	"time"

	"github.com/redis/go-redis/v9"
)

// MAEFEResult published to Redis after price tracking completes.
type MAEFEResult struct {
	Symbol    string  `json:"symbol"`
	Direction string  `json:"direction"`
	EntryPx   float64 `json:"entry_price"`
	MAEPct    float64 `json:"mae_pct"`
	MFEPct    float64 `json:"mfe_pct"`
	LowPx     float64 `json:"low_price"`
	HighPx    float64 `json:"high_price"`
}

type PriceTracker struct {
	mu       sync.Mutex
	trackers map[string]*trackedPosition
	rdb      *redis.Client
	channel  string
	logger   *slog.Logger
}

type trackedPosition struct {
	symbol    string
	direction string
	entryPx   float64
	startTime time.Time
	lowPx     float64
	highPx    float64
}

func NewPriceTracker(rdb *redis.Client, channel string, logger *slog.Logger) *PriceTracker {
	return &PriceTracker{
		trackers: make(map[string]*trackedPosition),
		rdb:      rdb,
		channel:  channel,
		logger:   logger,
	}
}

func (pt *PriceTracker) StartTracking(symbol, direction string, entryPx float64, duration time.Duration) {
	pt.mu.Lock()
	defer pt.mu.Unlock()

	pt.trackers[symbol] = &trackedPosition{
		symbol:    symbol,
		direction: direction,
		entryPx:   entryPx,
		startTime: time.Now(),
		lowPx:     entryPx,
		highPx:    entryPx,
	}

	go pt.trackLoop(symbol, duration)
}

func (pt *PriceTracker) trackLoop(symbol string, duration time.Duration) {
	ticker := time.NewTicker(5 * time.Second)
	defer ticker.Stop()
	endTime := time.Now().Add(duration)

	for {
		select {
		case <-ticker.C:
			pt.mu.Lock()
			_, ok := pt.trackers[symbol]
			pt.mu.Unlock()

			if !ok || time.Now().After(endTime) {
				pt.finishTracking(symbol)
				return
			}
		}
	}
}

func (pt *PriceTracker) UpdatePrice(symbol string, price float64) {
	pt.mu.Lock()
	defer pt.mu.Unlock()

	tp, ok := pt.trackers[symbol]
	if !ok || price <= 0 {
		return
	}

	if price < tp.lowPx {
		tp.lowPx = price
	}
	if price > tp.highPx {
		tp.highPx = price
	}
}

func (pt *PriceTracker) finishTracking(symbol string) {
	pt.mu.Lock()
	tp, ok := pt.trackers[symbol]
	if !ok {
		pt.mu.Unlock()
		return
	}
	delete(pt.trackers, symbol)
	pt.mu.Unlock()

	if tp.entryPx <= 0 {
		return
	}

	var mae, mfe float64
	if tp.direction == "LONG" {
		mae = (tp.entryPx - tp.lowPx) / tp.entryPx
		mfe = (tp.highPx - tp.entryPx) / tp.entryPx
	} else {
		mae = (tp.highPx - tp.entryPx) / tp.entryPx
		mfe = (tp.entryPx - tp.lowPx) / tp.entryPx
	}

	mae = math.Round(mae*10000) / 100
	mfe = math.Round(mfe*10000) / 100

	pt.logger.Info("MAE/MFE tracking complete",
		"symbol", symbol,
		"direction", tp.direction,
		"mae_pct", mae,
		"mfe_pct", mfe,
		"entry", tp.entryPx,
		"low", tp.lowPx,
		"high", tp.highPx,
	)

	result := MAEFEResult{
		Symbol:    symbol,
		Direction: tp.direction,
		EntryPx:   tp.entryPx,
		MAEPct:    mae,
		MFEPct:    mfe,
		LowPx:     tp.lowPx,
		HighPx:    tp.highPx,
	}

	data, err := json.Marshal(result)
	if err != nil {
		pt.logger.Error("failed to marshal MAE/MFE result", "err", err)
		return
	}

	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel()
	if err := pt.rdb.Publish(ctx, pt.channel, data).Err(); err != nil {
		pt.logger.Error("failed to publish MAE/MFE result", "channel", pt.channel, "err", err)
	}
}
