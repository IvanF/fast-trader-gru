package executor

import (
	"context"
)

// reconcileOrphanEntryOrders cancels untracked non-reduce-only limits left on the exchange
// after a container restart (infrastructure failsafe — no timers).
func (s *Service) reconcileOrphanEntryOrders(ctx context.Context) {
	orders, err := s.bybit.ListOpenOrders(ctx)
	if err != nil {
		s.logger.Warn("startup open-order scan failed", "error", err)
		return
	}

	s.mu.Lock()
	tracked := make(map[string]string, len(s.pending))
	for sym, p := range s.pending {
		tracked[sym] = p.OrderID
	}
	s.mu.Unlock()

	var cancelled int
	for _, o := range orders {
		if o.ReduceOnly || o.OrderID == "" {
			continue
		}
		if id, ok := tracked[o.Symbol]; ok && id == o.OrderID {
			continue
		}
		if err := s.bybit.CancelOrder(ctx, o.Symbol, o.OrderID); err != nil {
			s.logger.Warn("startup orphan entry cancel failed",
				"symbol", o.Symbol,
				"order_id", o.OrderID,
				"error", err,
			)
			continue
		}
		cancelled++
		s.logger.Warn("startup orphan entry cancelled",
			"symbol", o.Symbol,
			"order_id", o.OrderID,
			"side", o.Side,
			"price", o.Price,
		)
	}
	if cancelled > 0 {
		s.logger.Info("startup entry reconciliation complete", "cancelled", cancelled)
	}
}
