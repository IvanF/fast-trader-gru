package metrics

import (
	"net/http"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promhttp"
)

var (
	PointsWritten = prometheus.NewCounter(prometheus.CounterOpts{
		Name: "history_logger_points_written_total",
		Help: "Total InfluxDB points written",
	})
	PointsDropped = prometheus.NewCounter(prometheus.CounterOpts{
		Name: "history_logger_points_dropped_total",
		Help: "Points dropped due to full buffer",
	})
	BatchFlushes = prometheus.NewCounter(prometheus.CounterOpts{
		Name: "history_logger_batch_flushes_total",
		Help: "Number of batch flushes to InfluxDB",
	})
	RedisMessages = prometheus.NewCounterVec(prometheus.CounterOpts{
		Name: "history_logger_redis_messages_total",
		Help: "Redis messages received by type",
	}, []string{"type"})
	InfluxWriteErrors = prometheus.NewCounter(prometheus.CounterOpts{
		Name: "history_logger_influx_write_errors_total",
		Help: "InfluxDB write errors",
	})
	ParseErrors = prometheus.NewCounter(prometheus.CounterOpts{
		Name: "history_logger_parse_errors_total",
		Help: "Line protocol parse errors",
	})
	FlushDuration = prometheus.NewHistogram(prometheus.HistogramOpts{
		Name:    "history_logger_flush_duration_seconds",
		Help:    "Batch flush duration",
		Buckets: []float64{0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0},
	})
	BatchSize = prometheus.NewHistogram(prometheus.HistogramOpts{
		Name:    "history_logger_batch_size",
		Help:    "Points per flush batch",
		Buckets: []float64{100, 500, 1000, 5000, 10000, 20000},
	})
	BufferDepth = prometheus.NewGauge(prometheus.GaugeOpts{
		Name: "history_logger_buffer_depth",
		Help: "Current internal batch buffer depth",
	})
)

func init() {
	prometheus.MustRegister(
		PointsWritten, PointsDropped, BatchFlushes, RedisMessages,
		InfluxWriteErrors, ParseErrors, FlushDuration, BatchSize, BufferDepth,
	)
}

func Handler() http.Handler {
	return promhttp.Handler()
}
