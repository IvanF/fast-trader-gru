package metrics

import (
	"net/http"
	"sync"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promhttp"
)

var (
	TotalPnL = prometheus.NewGauge(prometheus.GaugeOpts{
		Name: "oms_total_pnl_usdt",
		Help: "Aggregate realized PnL in USDT",
	})
	SymbolPnL = prometheus.NewGaugeVec(prometheus.GaugeOpts{
		Name: "oms_symbol_pnl_usdt",
		Help: "Per-symbol realized PnL",
	}, []string{"symbol"})
	WinLossRatio = prometheus.NewGauge(prometheus.GaugeOpts{
		Name: "oms_win_loss_ratio",
		Help: "Win/loss ratio",
	})
	AvgHoldingTime = prometheus.NewGauge(prometheus.GaugeOpts{
		Name: "oms_avg_holding_time_seconds",
		Help: "Average holding time in seconds",
	})
	ActivePositions = prometheus.NewGauge(prometheus.GaugeOpts{
		Name: "oms_active_positions",
		Help: "Number of active positions",
	})
	GridActive = prometheus.NewGaugeVec(prometheus.GaugeOpts{
		Name: "oms_grid_active",
		Help: "Grid active status per symbol",
	}, []string{"symbol"})
	FundingRate = prometheus.NewGaugeVec(prometheus.GaugeOpts{
		Name: "oms_funding_rate",
		Help: "Current funding rate per symbol",
	}, []string{"symbol"})
	SignalToOrderLatency = prometheus.NewHistogram(prometheus.HistogramOpts{
		Name:    "oms_signal_to_order_latency_seconds",
		Help:    "Latency from signal receipt to order placement",
		Buckets: []float64{0.05, 0.1, 0.2, 0.5, 1.0, 2.0},
	})
	OrdersPlaced = prometheus.NewCounterVec(prometheus.CounterOpts{
		Name: "oms_orders_placed_total",
		Help: "Total orders placed",
	}, []string{"type"})
)

type PnLTracker struct {
	mu      sync.Mutex
	wins    int
	losses  int
	total   float64
	holdSum time.Duration
	trades  int
}

func (p *PnLTracker) Record(pnl float64, hold time.Duration) {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.total += pnl
	p.trades++
	p.holdSum += hold
	if pnl >= 0 {
		p.wins++
	} else {
		p.losses++
	}
	TotalPnL.Set(p.total)
	if p.losses > 0 {
		WinLossRatio.Set(float64(p.wins) / float64(p.losses))
	} else if p.wins > 0 {
		WinLossRatio.Set(float64(p.wins))
	}
	if p.trades > 0 {
		AvgHoldingTime.Set(p.holdSum.Seconds() / float64(p.trades))
	}
}

func init() {
	prometheus.MustRegister(
		TotalPnL, SymbolPnL, WinLossRatio, AvgHoldingTime,
		ActivePositions, GridActive, FundingRate,
		SignalToOrderLatency, OrdersPlaced,
	)
}

func Handler() http.Handler {
	return promhttp.Handler()
}
