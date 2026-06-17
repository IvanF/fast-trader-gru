package metrics

import (
	"net/http"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promhttp"
)

var (
	WSEventsTotal = prometheus.NewCounterVec(prometheus.CounterOpts{
		Name: "ingestion_ws_events_total",
		Help: "Total WebSocket events processed",
	}, []string{"type"})
	RedisPublishTotal = prometheus.NewCounterVec(prometheus.CounterOpts{
		Name: "ingestion_redis_publish_total",
		Help: "Total Redis publish operations",
	}, []string{"channel_type"})
	ActiveConnections = prometheus.NewGauge(prometheus.GaugeOpts{
		Name: "ingestion_active_ws_connections",
		Help: "Active Bybit WebSocket connections",
	})
	ReconnectTotal = prometheus.NewCounter(prometheus.CounterOpts{
		Name: "ingestion_ws_reconnect_total",
		Help: "WebSocket reconnection count",
	})
	PublishLatency = prometheus.NewHistogram(prometheus.HistogramOpts{
		Name:    "ingestion_redis_publish_latency_seconds",
		Help:    "Redis publish latency",
		Buckets: []float64{0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1},
	})
)

func init() {
	prometheus.MustRegister(WSEventsTotal, RedisPublishTotal, ActiveConnections, ReconnectTotal, PublishLatency)
}

func Handler() http.Handler {
	return promhttp.Handler()
}
