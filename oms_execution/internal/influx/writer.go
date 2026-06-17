package influx

import (
	"fmt"
	"log/slog"
	"strconv"
	"time"

	influxdb2 "github.com/influxdata/influxdb-client-go/v2"
	"github.com/influxdata/influxdb-client-go/v2/api"

	"github.com/fast-trader-gru/oms_execution/internal/models"
)

const tradeMeasurement = "trade_outcomes"

type Writer struct {
	writeAPI api.WriteAPI
	logger   *slog.Logger
}

func NewWriter(url, token, org, bucket string, logger *slog.Logger) (*Writer, error) {
	if token == "" {
		return nil, fmt.Errorf("INFLUX_TOKEN is required")
	}
	client := influxdb2.NewClient(url, token)
	writeAPI := client.WriteAPI(org, bucket)
	w := &Writer{writeAPI: writeAPI, logger: logger}
	go func() {
		for err := range writeAPI.Errors() {
			logger.Error("influx trade write error", "error", err)
		}
	}()
	return w, nil
}

func (w *Writer) WriteTradeOutcome(result models.ExecutionResult) {
	closedAt := result.ClosedAt
	if closedAt == 0 {
		closedAt = time.Now().UnixMilli()
	}
	ts := time.UnixMilli(closedAt)
	won := "false"
	if result.NetPnL >= 0 {
		won = "true"
	}
	exchangePnL := "false"
	if result.ExchangePnL {
		exchangePnL = "true"
	}

	line := fmt.Sprintf(
		"%s,symbol=%s,direction=%s,regime=%s,signal_id=%s,close_reason=%s net_pnl=%s,entry_price=%s,exit_price=%s,holding_time_ms=%si,partial_closed=%t,grid_levels=%si,won=%s,exchange_pnl=%s %d",
		tradeMeasurement,
		escapeTag(result.Symbol),
		escapeTag(result.Direction),
		escapeTag(result.Regime),
		escapeTag(result.SignalID),
		escapeTag(result.CloseReason),
		formatFloat(result.NetPnL),
		formatFloat(result.EntryPrice),
		formatFloat(result.ExitPrice),
		strconv.FormatInt(result.HoldingTimeMs, 10),
		result.PartialClosed,
		strconv.Itoa(result.GridLevels),
		won,
		exchangePnL,
		ts.UnixNano(),
	)
	w.writeAPI.WriteRecord(line)
}

func (w *Writer) Close() {
	w.writeAPI.Flush()
}

func escapeTag(s string) string {
	out := make([]byte, 0, len(s))
	for i := 0; i < len(s); i++ {
		switch s[i] {
		case ',', ' ', '=':
			out = append(out, '\\', s[i])
		default:
			out = append(out, s[i])
		}
	}
	return string(out)
}

func formatFloat(v float64) string {
	return strconv.FormatFloat(v, 'f', -1, 64)
}
