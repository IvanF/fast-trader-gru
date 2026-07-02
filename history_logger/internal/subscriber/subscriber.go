package subscriber

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"strings"
	"sync/atomic"
	"time"

	"github.com/fast-trader-gru/history_logger/internal/config"
	"github.com/fast-trader-gru/history_logger/internal/influx"
	"github.com/fast-trader-gru/history_logger/internal/lineproto"
	"github.com/fast-trader-gru/history_logger/internal/metrics"
	"github.com/fast-trader-gru/history_logger/internal/models"
	"github.com/redis/go-redis/v9"
	"github.com/vmihailenco/msgpack/v5"
)

type Service struct {
	cfg    config.Config
	rdb    *redis.Client
	writer *influx.BatchWriter
	logger *slog.Logger
	tracker *PriceTracker

	obMsgs uint64
	trMsgs uint64
}

func New(cfg config.Config, rdb *redis.Client, writer *influx.BatchWriter, logger *slog.Logger) *Service {
	return &Service{cfg: cfg, rdb: rdb, writer: writer, logger: logger, tracker: NewPriceTracker(rdb, "execution:mae_mfe", logger)}
}

func (s *Service) Run(ctx context.Context) error {
	go s.statsLoop(ctx)
	go s.listenExecutionResults(ctx)
	for {
		if ctx.Err() != nil {
			return ctx.Err()
		}
		pubsub := s.rdb.PSubscribe(ctx, "market:orderbook:*", "market:trades:*")
		ch := pubsub.Channel()
		s.logger.Info("subscribed to market channels")

		for msg := range ch {
			if ctx.Err() != nil {
				pubsub.Close()
				return ctx.Err()
			}
			s.handleMessage(msg.Channel, []byte(msg.Payload))
		}

		pubsub.Close()
		s.logger.Warn("redis subscription dropped, reconnecting")
		time.Sleep(2 * time.Second)
	}
}

func (s *Service) statsLoop(ctx context.Context) {
	ticker := time.NewTicker(60 * time.Second)
	defer ticker.Stop()
	var lastOB, lastTR uint64
	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			ob := atomic.LoadUint64(&s.obMsgs)
			tr := atomic.LoadUint64(&s.trMsgs)
			s.logger.Info("history_logger stats",
				"orderbook_msgs_1m", ob-lastOB,
				"trade_msgs_1m", tr-lastTR,
				"orderbook_total", ob,
				"trade_total", tr,
			)
			lastOB, lastTR = ob, tr
		}
	}
}

func (s *Service) handleMessage(channel string, raw []byte) {
	if strings.Contains(channel, "orderbook") {
		atomic.AddUint64(&s.obMsgs, 1)
		metrics.RedisMessages.WithLabelValues("orderbook").Inc()
		var ob models.OrderbookPayload
		if !decode(raw, &ob) {
			return
		}
		if ob.Symbol == "" {
			ob.Symbol = symbolFromChannel(channel)
		}
		for _, line := range lineproto.OrderbookDepthLines(ob, s.cfg.OrderbookDepth) {
			s.writer.Enqueue(line)
		}
		return
	}

	if strings.Contains(channel, "trades") {
		atomic.AddUint64(&s.trMsgs, 1)
		metrics.RedisMessages.WithLabelValues("trade").Inc()
		var trade models.TradePayload
		if !decode(raw, &trade) {
			return
		}
		if trade.Symbol == "" {
			trade.Symbol = symbolFromChannel(channel)
		}
		s.writer.Enqueue(lineproto.TradeLine(trade))
		s.tracker.UpdatePrice(trade.Symbol, trade.Price)
	}
}

func symbolFromChannel(channel string) string {
	parts := strings.Split(channel, ":")
	if len(parts) >= 3 {
		return parts[2]
	}
	return ""
}

func decode(raw []byte, dest any) bool {
	if err := msgpack.Unmarshal(raw, dest); err == nil {
		return true
	}
	return json.Unmarshal(raw, dest) == nil
}

// ExecutionResult represents a closed trade from OMS.
type ExecutionResult struct {
	SignalID    string  `json:"signal_id"`
	Symbol      string  `json:"symbol"`
	Direction   string  `json:"direction"`
	EntryPx     float64 `json:"entry_price"`
	ExitPx      float64 `json:"exit_price"`
	PnL         float64 `json:"net_pnl"`
	Reason      string  `json:"close_reason"`
	ExchangePnL bool    `json:"exchange_pnl"`
}

func (s *Service) listenExecutionResults(ctx context.Context) {
	pubsub := s.rdb.Subscribe(ctx, "execution:results")
	ch := pubsub.Channel()
	s.logger.Info("subscribed to execution:results for MAE/MFE tracking")

	for msg := range ch {
		if ctx.Err() != nil {
			pubsub.Close()
			return
		}
		var result ExecutionResult
		if !decode([]byte(msg.Payload), &result) {
			continue
		}
		if result.Symbol == "" || result.EntryPx <= 0 {
			continue
		}
		// Start MAE/MFE tracking for 30 minutes after trade close
		tradeID := result.SignalID
		if tradeID == "" {
			tradeID = result.Symbol
		}
		// Composite key: exchange and shadow trades share same SignalID
		tradeID = fmt.Sprintf("%s:%v", tradeID, result.ExchangePnL)
		s.tracker.StartTracking(tradeID, result.Symbol, result.Direction, result.EntryPx, 30*time.Minute)
		s.logger.Info("started MAE/MFE tracking",
			"trade_id", tradeID,
			"symbol", result.Symbol,
			"direction", result.Direction,
			"entry", result.EntryPx,
			"pnl", result.PnL,
		)
	}
}
