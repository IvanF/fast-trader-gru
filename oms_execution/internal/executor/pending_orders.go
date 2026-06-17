package executor

import (
	"context"
	"time"

	"github.com/fast-trader-gru/oms_execution/internal/models"
)

func (s *Service) publishPendingOrder(ctx context.Context, event string, p *models.PendingEntry) {
	if p == nil {
		return
	}
	evt := models.PendingOrderEvent{
		Event:      event,
		Symbol:     p.Symbol,
		Direction:  p.Direction,
		OrderID:    p.OrderID,
		EntryPrice: p.EntryPrice,
		Confidence: p.Signal.Confidence,
		SignalID:   p.Signal.SignalID,
		Timestamp:  time.Now().UnixMilli(),
	}
	if err := s.redis.Publish(ctx, s.cfg.PendingOrdersChannel, evt); err != nil {
		s.logger.Warn("pending order publish failed", "symbol", p.Symbol, "event", event, "error", err)
	}
}

func (s *Service) publishPendingCancelled(ctx context.Context, p *models.PendingEntry, reason string) {
	evt := models.PendingOrderEvent{
		Event:      "cancelled",
		Symbol:     p.Symbol,
		Direction:  p.Direction,
		OrderID:    p.OrderID,
		EntryPrice: p.EntryPrice,
		Confidence: p.Signal.Confidence,
		SignalID:   p.Signal.SignalID,
		Reason:     reason,
		Timestamp:  time.Now().UnixMilli(),
	}
	if err := s.redis.Publish(ctx, s.cfg.PendingOrdersChannel, evt); err != nil {
		s.logger.Warn("pending cancel publish failed", "symbol", p.Symbol, "error", err)
	}
}
