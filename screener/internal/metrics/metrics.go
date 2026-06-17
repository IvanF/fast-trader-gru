package metrics

import (
	"net/http"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promhttp"
)

var (
	ActiveSymbolsCount = prometheus.NewGauge(prometheus.GaugeOpts{
		Name: "screener_active_symbols_count",
		Help: "Number of symbols passing screener filters",
	})
	ScreenDuration = prometheus.NewHistogram(prometheus.HistogramOpts{
		Name:    "screener_screen_duration_seconds",
		Help:    "Duration of a full screener cycle",
		Buckets: prometheus.DefBuckets,
	})
	FundingRateFiltered = prometheus.NewCounter(prometheus.CounterOpts{
		Name: "screener_funding_rate_filtered_total",
		Help: "Symbols rejected due to extreme funding rate",
	})
	TurnoverFiltered = prometheus.NewCounter(prometheus.CounterOpts{
		Name: "screener_turnover_filtered_total",
		Help: "Symbols rejected due to low turnover",
	})
	BybitRequestErrors = prometheus.NewCounter(prometheus.CounterOpts{
		Name: "screener_bybit_request_errors_total",
		Help: "Bybit API request failures",
	})
)

func init() {
	prometheus.MustRegister(
		ActiveSymbolsCount,
		ScreenDuration,
		FundingRateFiltered,
		TurnoverFiltered,
		BybitRequestErrors,
	)
}

func Handler() http.Handler {
	return promhttp.Handler()
}
