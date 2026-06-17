package executor

import (
	"context"
	"time"

	"github.com/fast-trader-gru/oms_execution/internal/models"
)

func (s *Service) publishPositionOpened(ctx context.Context, pos *models.ActivePosition) {
	evt := models.PositionEvent{
		Event:      "opened",
		Symbol:     pos.Symbol,
		Direction:  pos.Direction,
		SignalID:   pos.Signal.SignalID,
		Confidence: pos.Signal.Confidence,
		EntryPrice: pos.FillPrice,
		Timestamp:  time.Now().UnixMilli(),
	}
	if err := s.redis.Publish(ctx, s.cfg.PositionsChannel, evt); err != nil {
		s.logger.Warn("position opened publish failed", "symbol", pos.Symbol, "error", err)
	}
}
