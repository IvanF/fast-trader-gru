package executor

import (
	"context"
	"log/slog"
	"math"
	"strconv"
	"sync"
	"time"

	"github.com/fast-trader-gru/oms_execution/internal/bybit"
	"github.com/fast-trader-gru/oms_execution/internal/models"
)

const (
	// LiquidityEvaporationThreshold: if wall volume drops below 25% of initial, cancel.
	LiquidityEvaporationThreshold = 0.25
	// MaxOrderAge: if order alive longer than this without fill, cancel (stale order).
	MaxOrderAge = 30 * time.Second
)

// TrackedOrder represents a passive Maker order being monitored.
type TrackedOrder struct {
	Symbol          string
	Price           float64
	Side            string
	InitialWallVol  float64
	PlacedAt        time.Time
}

// QueueMonitor watches orderbook depth for liquidity evaporation.
// If the wall protecting our order evaporates > 75%, we emergency-cancel.
type QueueMonitor struct {
	mu        sync.Mutex
	orders    map[string]*TrackedOrder
	cancelCh  chan string
	logger    *slog.Logger
	stats     map[string]int64
}

// NewQueueMonitor creates a new QueueMonitor with a background cancel loop.
func NewQueueMonitor(logger *slog.Logger) *QueueMonitor {
	qm := &QueueMonitor{
		orders:   make(map[string]*TrackedOrder),
		cancelCh: make(chan string, 100),
		logger:   logger,
		stats:    make(map[string]int64),
	}
	return qm
}

// StartCancelLoop runs in a goroutine — listens to cancelCh and sends cancel requests.
func (qm *QueueMonitor) StartCancelLoop(ctx context.Context, bybitClient *bybit.Client) {
	go func() {
		for {
			select {
			case <-ctx.Done():
				return
			case orderID := <-qm.cancelCh:
				qm.mu.Lock()
				order, exists := qm.orders[orderID]
				if !exists {
					qm.mu.Unlock()
					continue
				}
				delete(qm.orders, orderID)
				qm.mu.Unlock()

				if err := bybitClient.CancelOrder(ctx, order.Symbol, orderID); err != nil {
					qm.logger.Warn("queue monitor cancel failed",
						"order_id", orderID, "symbol", order.Symbol, "error", err)
					qm.stats["cancel_failed"]++
				} else {
					qm.logger.Warn("LIQUIDITY MIRAGE — emergency cancel",
						"order_id", orderID,
						"symbol", order.Symbol,
						"price", order.Price,
						"side", order.Side,
						"initial_wall", order.InitialWallVol,
						"held_for", time.Since(order.PlacedAt).String(),
					)
					qm.stats["cancel_success"]++
				}
			}
		}
	}()

	// Stale order cleanup: cancel orders older than MaxOrderAge
	go func() {
		ticker := time.NewTicker(5 * time.Second)
		defer ticker.Stop()
		for {
			select {
			case <-ctx.Done():
				return
			case <-ticker.C:
				now := time.Now()
				qm.mu.Lock()
				for orderID, order := range qm.orders {
					if now.Sub(order.PlacedAt) > MaxOrderAge {
						delete(qm.orders, orderID)
						qm.mu.Unlock()
						if err := bybitClient.CancelOrder(ctx, order.Symbol, orderID); err == nil {
							qm.logger.Info("stale order cancelled",
								"order_id", orderID, "symbol", order.Symbol,
								"held_for", now.Sub(order.PlacedAt).String())
							qm.stats["stale_cancel"]++
						}
						qm.mu.Lock()
					}
				}
				qm.mu.Unlock()
			}
		}
	}()
}

// MonitorOrder starts tracking an order for liquidity evaporation.
func (qm *QueueMonitor) MonitorOrder(orderID, symbol, side string, price, initialWallVol float64) {
	qm.mu.Lock()
	qm.orders[orderID] = &TrackedOrder{
		Symbol:         symbol,
		Price:          price,
		Side:           side,
		InitialWallVol: initialWallVol,
		PlacedAt:       time.Now(),
	}
	qm.mu.Unlock()
	qm.logger.Info("queue monitor: tracking order",
		"order_id", orderID, "symbol", symbol, "side", side,
		"price", price, "wall_vol", initialWallVol)
}

// RemoveOrder stops tracking an order (filled, cancelled, or expired).
func (qm *QueueMonitor) RemoveOrder(orderID string) {
	qm.mu.Lock()
	delete(qm.orders, orderID)
	qm.mu.Unlock()
}

// OnOrderbookUpdate checks if any tracked order's protective wall has evaporated.
func (qm *QueueMonitor) OnOrderbookUpdate(symbol string, bids, asks []models.OrderbookLevel) {
	qm.mu.Lock()
	defer qm.mu.Unlock()

	for orderID, order := range qm.orders {
		if order.Symbol != symbol {
			continue
		}

		var levels []models.OrderbookLevel
		if order.Side == "Buy" {
			levels = bids
		} else {
			levels = asks
		}

		currentVol := getVolumeAtPrice(order.Price, levels)

		// Liquidity Mirage: protective wall evaporated > 75%
		if order.InitialWallVol > 0 && currentVol < order.InitialWallVol*LiquidityEvaporationThreshold {
			qm.stats["mirage_detected"]++
			qm.cancelCh <- orderID
		}
	}
}

// Stats returns monitoring statistics.
func (qm *QueueMonitor) Stats() map[string]int64 {
	qm.mu.Lock()
	defer qm.mu.Unlock()
	result := make(map[string]int64, len(qm.stats))
	for k, v := range qm.stats {
		result[k] = v
	}
	result["active_orders"] = int64(len(qm.orders))
	return result
}

// ActiveOrders returns count of tracked orders.
func (qm *QueueMonitor) ActiveOrders() int {
	qm.mu.Lock()
	defer qm.mu.Unlock()
	return len(qm.orders)
}

// getVolumeAtPrice finds the orderbook level volume closest to target price.
func getVolumeAtPrice(targetPrice float64, levels []models.OrderbookLevel) float64 {
	bestVol := 0.0
	bestDist := math.MaxFloat64
	for _, lv := range levels {
		price, _ := strconv.ParseFloat(lv.Price, 64)
		size, _ := strconv.ParseFloat(lv.Size, 64)
		dist := math.Abs(price - targetPrice)
		if dist < bestDist {
			bestDist = dist
			bestVol = size
		}
	}
	return bestVol
}
