package executor

import (
	"context"
	"time"

	"github.com/fast-trader-gru/oms_execution/internal/bybit"
	"github.com/fast-trader-gru/oms_execution/internal/models"
)

func (s *Service) setGridDeploying(symbol string, deploying bool) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.gridDeploying == nil {
		s.gridDeploying = make(map[string]bool)
	}
	if deploying {
		s.gridDeploying[symbol] = true
	} else {
		delete(s.gridDeploying, symbol)
	}
}

func (s *Service) isGridDeploying(symbol string) bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.gridDeploying != nil && s.gridDeploying[symbol]
}

// pollPositionAfterCancel retries GetPosition after a cancel attempt to catch
// ghost fills (order gone on exchange before position API reflects the fill).
func (s *Service) pollPositionAfterCancel(ctx context.Context, symbol, direction string, cancelErr error) *bybit.PositionInfo {
	attempts := 1
	pause := 250 * time.Millisecond
	if cancelErr != nil && bybit.IsOrderNotCancelable(cancelErr) {
		attempts = 5
	} else if cancelErr != nil {
		attempts = 3
	}

	for i := 0; i < attempts; i++ {
		if i > 0 {
			select {
			case <-ctx.Done():
				return nil
			case <-time.After(pause):
			}
		}
		exPos, err := s.bybit.GetPosition(ctx, symbol)
		if err != nil || exPos.Size <= 0 || !s.positionMatchesDirection(exPos, direction) {
			continue
		}
		return &exPos
	}
	return nil
}

func (s *Service) tryPromotePendingFromExchange(ctx context.Context, p *models.PendingEntry, reason string, exPos *bybit.PositionInfo) bool {
	if exPos == nil || exPos.Size <= 0 {
		return false
	}
	avgPrice := exPos.AvgPrice
	if avgPrice <= 0 {
		avgPrice = p.EntryPrice
	}
	qty := bybit.NormalizeQty(exPos.Size, p.QtyStep, p.MinOrderQty)
	s.mu.Lock()
	_, still := s.pending[p.Symbol]
	s.mu.Unlock()
	if !still || qty <= 0 {
		return false
	}
	s.logger.Info("ghost fill detected — promoting executed qty",
		"symbol", p.Symbol,
		"reason", reason,
		"qty", qty,
		"order_id", p.OrderID,
	)
	s.promotePending(ctx, p, avgPrice, qty)
	return true
}
