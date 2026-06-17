package executor

import (
	"context"

	"github.com/fast-trader-gru/oms_execution/internal/models"
)

// cleanupSymbolOrdersAfterClose cancels tracked exit/pending orders and removes
// all remaining open limits on the exchange for the symbol.
func (s *Service) cleanupSymbolOrdersAfterClose(ctx context.Context, symbol string, pos *models.ActivePosition) {
	if pos != nil {
		s.cancelExitOrders(ctx, pos)
	}
	s.dropPendingEntry(ctx, symbol, "position_closed")

	if err := s.bybit.CancelAllOrders(ctx, symbol); err != nil {
		s.logger.Warn("cleanup cancel-all failed",
			"symbol", symbol,
			"error", err,
		)
		return
	}
	s.logger.Info("symbol order cleanup complete", "symbol", symbol)
}

func (s *Service) dropPendingEntry(ctx context.Context, symbol, reason string) {
	s.mu.Lock()
	p := s.pending[symbol]
	delete(s.pending, symbol)
	s.mu.Unlock()
	if p == nil || p.OrderID == "" {
		return
	}
	if err := s.bybit.CancelOrder(ctx, symbol, p.OrderID); err != nil {
		s.logger.Warn("cleanup pending entry cancel failed",
			"symbol", symbol,
			"order_id", p.OrderID,
			"reason", reason,
			"error", err,
		)
		return
	}
	s.logger.Info("cleanup cancelled pending entry",
		"symbol", symbol,
		"order_id", p.OrderID,
		"reason", reason,
	)
}
