package executor

import (
	"context"
	"fmt"
	"time"

	"github.com/fast-trader-gru/oms_execution/internal/bybit"
	"github.com/fast-trader-gru/oms_execution/internal/models"
)

// syncPositionFromExchange updates local qty/price from Bybit and returns live size.
func (s *Service) syncPositionFromExchange(ctx context.Context, pos *models.ActivePosition) (float64, bool, error) {
	exPos, err := s.bybit.GetPosition(ctx, pos.Symbol)
	if err != nil {
		return 0, false, err
	}
	if exPos.Size <= 0 || !s.positionMatchesDirection(exPos, pos.Direction) {
		return 0, false, nil
	}
	size := bybit.NormalizeQty(exPos.Size, pos.QtyStep, pos.MinOrderQty)
	if size <= 0 {
		return 0, false, nil
	}
	if exPos.AvgPrice > 0 {
		pos.FillPrice = exPos.AvgPrice
	}
	pos.RemainingQty = size
	pos.InitialQty = size
	return size, true, nil
}

func (s *Service) openTPQty(pos *models.ActivePosition) float64 {
	var sum float64
	for i := range pos.TakeProfitOrders {
		tp := &pos.TakeProfitOrders[i]
		if tp.Filled {
			continue
		}
		sum += tp.Qty
	}
	return sum
}

// ensureExchangeFlat closes any remaining exchange position with reduce-only market orders.
func (s *Service) ensureExchangeFlat(ctx context.Context, pos *models.ActivePosition, reason string) error {
	for attempt := 0; attempt < maxMarketCloseRetries; attempt++ {
		if ctx.Err() != nil {
			return ctx.Err()
		}
		exSize, hasPos, err := s.syncPositionFromExchange(ctx, pos)
		if err != nil {
			return err
		}
		if !hasPos || exSize <= 0 {
			return nil
		}
		s.logger.Warn("flattening remainder",
			"symbol", pos.Symbol,
			"reason", reason,
			"qty", exSize,
			"attempt", attempt+1,
		)
		side := closeSide(pos.Direction)
		_, err = s.bybit.PlaceReduceMarketRetry(ctx, pos.Symbol, side, exSize, pos.QtyStep)
		if err != nil {
			if bybit.IsRateLimitError(err) {
				continue
			}
			return fmt.Errorf("reduce market close: %w", err)
		}
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-time.After(350 * time.Millisecond):
		}
	}
	exSize, hasPos, err := s.syncPositionFromExchange(ctx, pos)
	if err != nil {
		return err
	}
	if hasPos && exSize > 0 {
		return fmt.Errorf("position still open after flatten attempts: %.8f", exSize)
	}
	return nil
}

const maxMarketCloseRetries = 6
