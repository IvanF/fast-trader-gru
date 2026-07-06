package executor

import (
	"context"
	"fmt"
	"time"

	"github.com/fast-trader-gru/oms_execution/internal/bybit"
	"github.com/fast-trader-gru/oms_execution/internal/models"
)

type entryOrderOutcome int

const (
	entryOrderDead entryOrderOutcome = iota
	entryOrderExecuted
)

func entryOrderIsDead(status string) bool {
	switch status {
	case "Cancelled", "Rejected", "Deactivated":
		return true
	default:
		return false
	}
}

func entryOrderIsExecuted(status string, cumExecQty float64) bool {
	if status == "Filled" {
		return true
	}
	return status == "PartiallyFilled" && cumExecQty > 0
}

// cancelEntryOrderConfirmed cancels a pending entry and polls until the order is dead or executed.
func (s *Service) cancelEntryOrderConfirmed(ctx context.Context, p *models.PendingEntry, orderID string) (entryOrderOutcome, error) {
	cancelErr := s.bybit.CancelOrder(ctx, p.Symbol, orderID)
	if cancelErr != nil && !bybit.IsOrderNotCancelable(cancelErr) {
		return entryOrderDead, cancelErr
	}

	const polls = 10
	for i := 0; i < polls; i++ {
		if ctx.Err() != nil {
			return entryOrderDead, ctx.Err()
		}

		order, err := s.bybit.GetOrderRealtime(ctx, p.Symbol, orderID)
		if err != nil {
			return entryOrderDead, err
		}

		if entryOrderIsExecuted(order.OrderStatus, order.CumExecQty) {
			return entryOrderExecuted, nil
		}
		if entryOrderIsDead(order.OrderStatus) {
			return entryOrderDead, nil
		}
		if order.OrderID == "" {
			if s.exchangeHasEntryFill(ctx, p) {
				return entryOrderExecuted, nil
			}
			return entryOrderDead, nil
		}

		if cancelErr == nil && order.OrderStatus == "New" {
			// Cancel accepted; wait for terminal state.
		}

		select {
		case <-ctx.Done():
			return entryOrderDead, ctx.Err()
		case <-time.After(120 * time.Millisecond):
		}
	}

	order, err := s.bybit.GetOrderRealtime(ctx, p.Symbol, orderID)
	if err != nil {
		return entryOrderDead, err
	}
	if entryOrderIsExecuted(order.OrderStatus, order.CumExecQty) {
		return entryOrderExecuted, nil
	}
	if entryOrderIsDead(order.OrderStatus) {
		return entryOrderDead, nil
	}
	if s.exchangeHasEntryFill(ctx, p) {
		return entryOrderExecuted, nil
	}
	return entryOrderDead, fmt.Errorf("entry order %s still active after cancel poll", orderID)
}

func (s *Service) exchangeHasEntryFill(ctx context.Context, p *models.PendingEntry) bool {
	exPos, err := s.bybit.GetPosition(ctx, p.Symbol)
	if err != nil || exPos.Size <= 0 {
		return false
	}
	return s.positionMatchesDirection(exPos, p.Direction)
}

// promotePendingFromExchange accepts an exchange fill and deploys the exit grid.
// It atomically claims the pending entry to prevent orphan scan from double-adopting.
func (s *Service) promotePendingFromExchange(ctx context.Context, p *models.PendingEntry, reason string) bool {
	s.mu.Lock()
	if cur, ok := s.pending[p.Symbol]; !ok || cur != p {
		s.mu.Unlock()
		return false
	}
	// Atomically remove pending entry under the same lock to prevent orphan scan race.
	// Mark as "promoting" so orphan scan skips this symbol during the async gap.
	delete(s.pending, p.Symbol)
	s._promoting[p.Symbol] = true
	s.mu.Unlock()

	exPos, err := s.bybit.GetPosition(ctx, p.Symbol)
	if err != nil || exPos.Size <= 0 || !s.positionMatchesDirection(exPos, p.Direction) {
		s.mu.Lock()
		delete(s.pending, p.Symbol)
		s.mu.Unlock()
		return false
	}

	avgPrice := exPos.AvgPrice
	if avgPrice <= 0 {
		avgPrice = p.EntryPrice
	}
	qty := bybit.NormalizeQty(exPos.Size, p.QtyStep, p.MinOrderQty)
	if qty <= 0 {
		return false
	}

	s.logger.Warn("reprice halted — entry order filled, promoting position",
		"symbol", p.Symbol,
		"reason", reason,
		"qty", qty,
		"target_qty", p.Qty,
		"order_id", p.OrderID,
	)
	s.promotePending(ctx, p, avgPrice, qty)
	return true
}

func (s *Service) replacePendingEntry(
	ctx context.Context,
	p *models.PendingEntry,
	signal models.TradeSignal,
	plan models.GridPlan,
	inst bybit.InstrumentInfo,
	ob models.OrderbookSnapshot,
	reason string,
) error {
	s.mu.Lock()
	if cur, ok := s.pending[p.Symbol]; !ok || cur != p {
		s.mu.Unlock()
		return nil
	}
	if p.State == models.PendingEntryStateCancelling {
		s.mu.Unlock()
		s.logger.Debug("reprice blocked — cancel in flight", "symbol", p.Symbol)
		return nil
	}
	oldOrderID := p.OrderID
	s.mu.Unlock()

	if !s.beginPendingEntryCancel(p) {
		s.logger.Debug("reprice blocked — cancel already in flight", "symbol", p.Symbol)
		return nil
	}
	defer s.releasePendingEntryCancelIfStuck(p)

	outcome, err := s.cancelEntryOrderConfirmed(ctx, p, oldOrderID)
	if err != nil {
		s.logger.Warn("cancel before reprice failed", "symbol", p.Symbol, "order_id", oldOrderID, "error", err)
		if s.promotePendingFromExchange(ctx, p, "reprice_cancel_ambiguous") {
			return nil
		}
		return err
	}
	if outcome == entryOrderExecuted {
		s.promotePendingFromExchange(ctx, p, reason+"_filled_on_cancel")
		return nil
	}

	s.mu.Lock()
	cur, ok := s.pending[p.Symbol]
	if !ok || cur != p {
		s.mu.Unlock()
		return nil
	}
	_, hasPos := s.positions[p.Symbol]
	s.mu.Unlock()
	if hasPos {
		s.mu.Lock()
		delete(s.pending, p.Symbol)
		s.mu.Unlock()
		return nil
	}

	side := "Buy"
	if plan.Direction == "SHORT" {
		side = "Sell"
	}
	orderID, err := s.bybit.PlaceLimitOrder(ctx, bybit.PlaceOrderRequest{
		Symbol:      plan.Symbol,
		Side:        side,
		Qty:         bybit.FormatQty(p.Qty, inst.Lot.QtyStep),
		Price:       bybit.FormatPrice(plan.EntryPrice),
		PositionIdx: 0,
	})
	if err != nil {
		if s.promotePendingFromExchange(ctx, p, "reprice_place_failed") {
			return nil
		}
		s.mu.Lock()
		delete(s.pending, p.Symbol)
		s.mu.Unlock()
		s.publishPendingCancelled(ctx, p, "reprice_failed")
		return err
	}

	s.mu.Lock()
	if cur, ok := s.pending[p.Symbol]; !ok || cur != p {
		s.mu.Unlock()
		_ = s.bybit.CancelOrder(ctx, p.Symbol, orderID)
		return nil
	}
	p.OrderID = orderID
	p.EntryPrice = plan.EntryPrice
	p.StopLoss = plan.StopLoss
	p.TakeProfits = plan.TakeProfits
	p.Signal = signal
	p.Orderbook = ob
	p.PlacedAt = time.Now().UnixMilli()
	p.State = models.PendingEntryStateActive
	s.mu.Unlock()

	s.publishPendingOrder(ctx, "repriced", p)

	s.logger.Info("pending entry repriced",
		"symbol", p.Symbol,
		"reason", reason,
		"new_entry", plan.EntryPrice,
		"order_id", orderID,
	)
	return nil
}
