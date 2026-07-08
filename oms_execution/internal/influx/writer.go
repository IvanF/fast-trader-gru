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
const gkMeasurement = "gatekeeper_features"

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

func (w *Writer) WriteGatekeeperFeatures(pos *models.ActivePosition, netPnL float64, recentWR float64) {
	if pos == nil {
		return
	}

	ts := time.Now()
	won := 0
	if netPnL > 0 {
		won = 1
	}

	tags := map[string]string{
		"symbol":       pos.Symbol,
		"direction":    pos.Direction,
		"close_reason": pos.Signal.ExitReason,
	}

	fields := map[string]interface{}{
		"confidence":            pos.Signal.Confidence,
		"pred_pnl":              0.0,
		"spread_pct":            pos.SpreadPctAtEntry,
		"obi":                   pos.OBIAtEntry,
		"volume_ratio":          pos.VolumeRatioAtEntry,
		"momentum":              pos.MomentumAtEntry,
		"price_velocity":        pos.PriceVelocityAtEntry,
		"atr_pct":               pos.ATRPctAtEntry,
		"funding_rate":          pos.Signal.FundingRate,
		"regime":                pos.Signal.Regime,
		"btc_correlation":       pos.Signal.BTCorrelation,
		"volatility_multiplier": pos.Signal.VolatilityMultiplier,
		"symbol_wr":             0.0,
		"symbol_pnl_sum":        0.0,
		"symbol_consec_losses":  0,
		"symbol_trades_24h":     0,
		"hour_of_day":           ts.UTC().Hour(),
		"day_of_week":           int(ts.UTC().Weekday()),
		"open_positions_count":  pos.OpenPositionsAtEntry,
		"recent_wr_20":          recentWR,
		"net_pnl":               netPnL,
		"label":                 won,
	}

	line := fmt.Sprintf(
		"%s,symbol=%s,direction=%s,close_reason=%s confidence=%s,spread_pct=%s,obi=%s,volume_ratio=%s,momentum=%s,price_velocity=%s,atr_pct=%s,funding_rate=%s,btc_correlation=%s,volatility_multiplier=%s,recent_wr_20=%s,net_pnl=%s,label=%di,hour_of_day=%di,day_of_week=%di,open_positions_count=%di %d",
		gkMeasurement,
		escapeTag(tags["symbol"]),
		escapeTag(tags["direction"]),
		escapeTag(tags["close_reason"]),
		formatFloat(fields["confidence"].(float64)),
		formatFloat(fields["spread_pct"].(float64)),
		formatFloat(fields["obi"].(float64)),
		formatFloat(fields["volume_ratio"].(float64)),
		formatFloat(fields["momentum"].(float64)),
		formatFloat(fields["price_velocity"].(float64)),
		formatFloat(fields["atr_pct"].(float64)),
		formatFloat(fields["funding_rate"].(float64)),
		formatFloat(fields["btc_correlation"].(float64)),
		formatFloat(fields["volatility_multiplier"].(float64)),
		formatFloat(recentWR),
		formatFloat(netPnL),
		won,
		fields["hour_of_day"].(int),
		fields["day_of_week"].(int),
		fields["open_positions_count"].(int),
		ts.UnixNano(),
	)
	w.writeAPI.WriteRecord(line)
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
