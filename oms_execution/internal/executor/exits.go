package executor

import (
	"context"
	"fmt"
	"math"
	"time"

	"github.com/fast-trader-gru/oms_execution/internal/bybit"
	"github.com/fast-trader-gru/oms_execution/internal/grid"
	"github.com/fast-trader-gru/oms_execution/internal/liquidity"
	"github.com/fast-trader-gru/oms_execution/internal/metrics"
	"github.com/fast-trader-gru/oms_execution/internal/models"
)

func closeSide(direction string) string {
	if direction == "LONG" {
		return "Sell"
	}
	return "Buy"
}

func hasRemainingSize(size, minQty float64) bool {
	return size >= minQty*0.99
}

func (s *Service) enforceSLPrice(ctx context.Context, pos *models.ActivePosition, slPrice, tickSize float64) float64 {
	minSLPct := s.cfg.GetMinSLPct(pos.Symbol)
	adjusted := grid.EnforceMinSLDistance(pos.FillPrice, slPrice, pos.Direction, minSLPct, tickSize)
	if grid.SLDistancePct(pos.FillPrice, adjusted) < minSLPct {
		s.logger.Info("sl widened to min distance",
			"symbol", pos.Symbol,
			"fill", pos.FillPrice,
			"requested", slPrice,
			"adjusted", adjusted,
			"min_pct", minSLPct,
		)
	}
	ob, err := s.redis.GetOrderbook(ctx, pos.Symbol)
	if err == nil {
		mid := grid.MidPrice(ob)
		if mid > 0 {
			minDist := pos.FillPrice * minSLPct
			if pos.Direction == "SHORT" && adjusted <= mid {
				adjusted = grid.RoundToTick(mid+minDist, tickSize)
				s.logger.Info("sl moved above current price (underwater)",
					"symbol", pos.Symbol, "mid", mid, "adjusted", adjusted)
			}
			if pos.Direction == "LONG" && adjusted >= mid {
				adjusted = grid.RoundToTick(mid-minDist, tickSize)
				s.logger.Info("sl moved below current price (underwater)",
					"symbol", pos.Symbol, "mid", mid, "adjusted", adjusted)
			}
		}
	}
	return adjusted
}

func (s *Service) deployExitGrid(ctx context.Context, pos *models.ActivePosition, ob models.OrderbookSnapshot, plannedEntry, plannedSL float64, tickSize float64) error {
	s.setGridDeploying(pos.Symbol, true)
	defer s.setGridDeploying(pos.Symbol, false)

	s.deployMu.Lock()
	defer s.deployMu.Unlock()

	s.cancelExitOrders(ctx, pos)
	_ = s.bybit.CancelAllOrders(ctx, pos.Symbol)
	_ = s.bybit.CancelAllConditionalOrders(ctx, pos.Symbol)

	exSize, hasPos, err := s.syncPositionFromExchange(ctx, pos)
	if err != nil {
		return fmt.Errorf("sync position: %w", err)
	}
	if !hasPos || !hasRemainingSize(exSize, pos.MinOrderQty) {
		return fmt.Errorf("no exchange position for exit grid")
	}

	tpRefPrice := pos.FillPrice
	opts := s.exitGridOptsForSymbol(pos.Symbol)
	opts.MaxTPPct = 0
	exitGrid := grid.BuildExitGrid(
		pos.Direction,
		tpRefPrice,
		plannedEntry,
		plannedSL,
		ob,
		pos.Signal,
		tickSize,
		exSize,
		pos.QtyStep,
		pos.MinOrderQty,
		opts,
	)
	slPrice := s.enforceSLPrice(ctx, pos, exitGrid.StopLoss.Price, tickSize)
	side := closeSide(pos.Direction)

	// Place TPs first (reduce-only limits), then SL covers 100% of remaining exposure.
	pos.TakeProfitOrders = nil
	for _, tp := range exitGrid.TakeProfits {
		if tp.Qty <= 0 {
			continue
		}
		tpID, err := s.bybit.PlaceReduceLimit(ctx, pos.Symbol, side, tp.Qty, pos.QtyStep, bybit.FormatPrice(tp.Price))
		if err != nil {
			s.logger.Warn("tp limit failed", "symbol", pos.Symbol, "kind", tp.Kind, "price", tp.Price, "qty", tp.Qty, "error", err)
			continue
		}
		pos.TakeProfitOrders = append(pos.TakeProfitOrders, models.ExitOrder{
			OrderID: tpID,
			Price:   tp.Price,
			Qty:     tp.Qty,
			Kind:    tp.Kind,
		})
		metrics.OrdersPlaced.WithLabelValues("take_profit").Inc()
	}

	exSize, hasPos, err = s.syncPositionFromExchange(ctx, pos)
	if err != nil {
		return err
	}
	if !hasPos || !hasRemainingSize(exSize, pos.MinOrderQty) {
		if len(pos.TakeProfitOrders) > 0 {
			pos.ExitGridReady = true
			return nil
		}
		return fmt.Errorf("position vanished after tp placement")
	}

	// SL covers only the remainder not allocated to open TP limits (not full exSize).
	slQty := exitGrid.StopLoss.Qty
	if slQty <= 0 {
		var tpPlaced float64
		for _, tp := range pos.TakeProfitOrders {
			tpPlaced += tp.Qty
		}
		slQty = exSize - tpPlaced
	}
	if slQty <= 0 {
		pos.ExitGridReady = true
		pos.LastGridDeployAt = time.Now().UnixMilli()
		s.logger.Info("exit grid deployed (tp-only, no sl remainder)",
			"symbol", pos.Symbol,
			"tp_levels", len(pos.TakeProfitOrders),
			"ex_size", exSize,
		)
		return nil
	}
	slQty = bybit.NormalizeQty(slQty, pos.QtyStep, pos.MinOrderQty)
	if slQty > exSize {
		slQty = bybit.NormalizeQty(exSize, pos.QtyStep, pos.MinOrderQty)
	}
	if slQty <= 0 {
		pos.ExitGridReady = true
		pos.LastGridDeployAt = time.Now().UnixMilli()
		s.logger.Info("exit grid deployed (tp-only, no sl remainder)",
			"symbol", pos.Symbol,
			"tp_levels", len(pos.TakeProfitOrders),
			"ex_size", exSize,
		)
		return nil
	}
	if err := s.placeStopLossCoveringRemainder(ctx, pos, slPrice, slQty, tickSize, "stop_loss"); err != nil {
		return err
	}

	pos.StopLoss = slPrice
	pos.PlannedSL = slPrice
	pos.ExitGridReady = true
	pos.LastGridDeployAt = time.Now().UnixMilli()

	s.logger.Info("exit grid deployed",
		"symbol", pos.Symbol,
		"sl", slPrice,
		"sl_qty", slQty,
		"tp_levels", len(pos.TakeProfitOrders),
		"fill", pos.FillPrice,
		"ex_size", exSize,
		"regime", pos.Signal.Regime,
	)
	for _, tp := range pos.TakeProfitOrders {
		s.logger.Info("tp order placed",
			"symbol", pos.Symbol,
			"kind", tp.Kind,
			"price", tp.Price,
			"qty", tp.Qty,
			"order_id", tp.OrderID,
		)
	}
	return nil
}

func (s *Service) placeStopLossCoveringRemainder(
	ctx context.Context,
	pos *models.ActivePosition,
	price, qty float64,
	tickSize float64,
	kind string,
) error {
	price = s.enforceSLPrice(ctx, pos, price, tickSize)
	qty = bybit.NormalizeQty(qty, pos.QtyStep, pos.MinOrderQty)
	if qty <= 0 {
		return fmt.Errorf("sl qty zero")
	}
	side := closeSide(pos.Direction)
	triggerDir := 2
	if pos.Direction == "SHORT" {
		triggerDir = 1
	}
	slID, err := s.bybit.PlaceStopMarket(ctx, pos.Symbol, side, qty, pos.QtyStep, bybit.FormatPrice(price), triggerDir)
	if err != nil {
		return fmt.Errorf("sl stop-market: %w", err)
	}
	pos.StopLossOrder = &models.ExitOrder{
		OrderID: slID,
		Price:   price,
		Qty:     qty,
		Kind:    kind,
		IsStop:  true,
	}
	metrics.OrdersPlaced.WithLabelValues("stop_loss").Inc()
	s.logger.Info("sl order placed",
		"symbol", pos.Symbol,
		"price", price,
		"qty", qty,
		"kind", kind,
		"order_id", slID,
	)
	return nil
}

// atomicReplaceStopLoss places the new SL first, verifies it, then cancels the old SL.
func (s *Service) atomicReplaceStopLoss(ctx context.Context, pos *models.ActivePosition, price, qty float64, kind string) error {
	skipEnforce := kind == "confidence_decay_exit" || kind == "signal_exit" || kind == "trailing_stop"
	if skipEnforce {
		price = grid.RoundToTick(price, pos.TickSize)
	} else {
		price = s.enforceSLPrice(ctx, pos, price, pos.TickSize)
	}
	if pos.StopLoss > 0 && kind == "stop_loss" && slWouldWiden(pos.Direction, pos.StopLoss, price) {
		s.logger.Warn("sl widen blocked",
			"symbol", pos.Symbol,
			"direction", pos.Direction,
			"current", pos.StopLoss,
			"requested", price,
			"kind", kind,
		)
		return nil
	}
	// Validate SL price is on correct side of current price
	ob, obErr := s.redis.GetOrderbook(ctx, pos.Symbol)
	if obErr == nil {
		mid := grid.MidPrice(ob)
		if mid > 0 {
			minDist := pos.FillPrice * s.cfg.GetMinSLPct(pos.Symbol)
			if pos.Direction == "SHORT" && price <= mid {
				price = grid.RoundToTick(mid+minDist, pos.TickSize)
				s.logger.Info("sl moved above current price (underwater)",
					"symbol", pos.Symbol, "mid", mid, "adjusted", price)
			}
			if pos.Direction == "LONG" && price >= mid {
				price = grid.RoundToTick(mid-minDist, pos.TickSize)
				s.logger.Info("sl moved below current price (underwater)",
					"symbol", pos.Symbol, "mid", mid, "adjusted", price)
			}
		}
	}
	qty = bybit.NormalizeQty(qty, pos.QtyStep, pos.MinOrderQty)
	if qty <= 0 {
		return nil
	}
	if pos.StopLossOrder != nil && !pos.StopLossOrder.Filled && qty < pos.StopLossOrder.Qty && kind == "stop_loss" {
		return nil
	}

	old := pos.StopLossOrder
	side := closeSide(pos.Direction)
	var newID string
	var err error
	isMaker := kind == "confidence_decay_exit" || kind == "signal_exit"
	if isMaker {
		newID, err = s.bybit.PlaceReducePostOnlyLimit(ctx, pos.Symbol, side, qty, pos.QtyStep, bybit.FormatPrice(price))
	} else {
		triggerDir := 2
		if pos.Direction == "SHORT" {
			triggerDir = 1
		}
		newID, err = s.bybit.PlaceStopMarket(ctx, pos.Symbol, side, qty, pos.QtyStep, bybit.FormatPrice(price), triggerDir)
	}
	if err != nil {
		return fmt.Errorf("atomic sl place: %w", err)
	}

	info, err := s.bybit.GetOrderRealtime(ctx, pos.Symbol, newID)
	if err != nil || info.OrderID == "" || info.OrderStatus == "Rejected" || info.OrderStatus == "Cancelled" {
		_ = s.bybit.CancelOrder(ctx, pos.Symbol, newID)
		return fmt.Errorf("atomic sl verify failed: %v status=%s", err, info.OrderStatus)
	}

	if old != nil && !old.Filled && old.OrderID != "" && old.OrderID != newID {
		if old.IsStop {
			if err := s.bybit.CancelStopOrder(ctx, pos.Symbol, old.OrderID); err != nil {
				s.logger.Warn("atomic sl cancel old failed", "symbol", pos.Symbol, "order_id", old.OrderID, "error", err)
			}
		} else {
			if err := s.bybit.CancelOrder(ctx, pos.Symbol, old.OrderID); err != nil {
				s.logger.Warn("atomic sl cancel old failed", "symbol", pos.Symbol, "order_id", old.OrderID, "error", err)
			}
		}
	}

	pos.StopLoss = price
	pos.StopLossOrder = &models.ExitOrder{
		OrderID: newID,
		Price:   price,
		Qty:     qty,
		Kind:    kind,
		IsStop:  kind != "confidence_decay_exit" && kind != "signal_exit",
	}
	s.logger.Info("stop loss atomically replaced",
		"symbol", pos.Symbol,
		"price", price,
		"qty", qty,
		"kind", kind,
		"order_id", newID,
	)
	return nil
}

func (s *Service) replaceStopLoss(ctx context.Context, pos *models.ActivePosition, price, qty float64) error {
	return s.atomicReplaceStopLoss(ctx, pos, price, qty, "stop_loss")
}

// slCoverQty is the SL size: exchange position minus qty already allocated to open TP limits.
func (s *Service) slCoverQty(pos *models.ActivePosition, exSize float64) float64 {
	tpOpen := s.openTPQty(pos)
	qty := exSize - tpOpen
	if qty <= pos.MinOrderQty*0.99 {
		return 0
	}
	return bybit.NormalizeQty(qty, pos.QtyStep, pos.MinOrderQty)
}

func (s *Service) syncSLToFullRemainder(ctx context.Context, pos *models.ActivePosition) {
	if pos.LastGridDeployAt > 0 && time.Now().UnixMilli()-pos.LastGridDeployAt < 2000 {
		return
	}
	exSize, hasPos, err := s.syncPositionFromExchange(ctx, pos)
	if err != nil || !hasPos || exSize <= 0 {
		return
	}
	if pos.StopLossOrder == nil || pos.StopLossOrder.Filled {
		return
	}
	slQty := s.slCoverQty(pos, exSize)
	if slQty <= 0 {
		return
	}
	if slQty <= pos.StopLossOrder.Qty && pos.StopLossOrder.Kind == "stop_loss" {
		return
	}
	price := pos.StopLoss
	if price <= 0 {
		price = pos.StopLossOrder.Price
	}
	ob, obErr := s.redis.GetOrderbook(ctx, pos.Symbol)
	if obErr == nil {
		mid := grid.MidPrice(ob)
		if mid > 0 {
			if pos.Direction == "SHORT" && price <= mid {
				return
			}
			if pos.Direction == "LONG" && price >= mid {
				return
			}
		}
	}
	if err := s.atomicReplaceStopLoss(ctx, pos, price, slQty, "stop_loss"); err != nil {
		s.logger.Warn("sync sl to full remainder failed", "symbol", pos.Symbol, "error", err)
	}
}

func (s *Service) refreshExitOrder(ctx context.Context, symbol string, order *models.ExitOrder) {
	if order == nil || order.Filled || order.OrderID == "" {
		return
	}
	info, err := s.bybit.GetOrderRealtime(ctx, symbol, order.OrderID)
	if err != nil {
		return
	}
	if info.OrderStatus == "Filled" || info.OrderStatus == "PartiallyFilled" {
		order.FilledQty = info.CumExecQty
		if info.AvgPrice > 0 {
			order.FilledPx = info.AvgPrice
		}
	}
	if info.OrderStatus == "Filled" {
		order.Filled = true
	}
}

func (s *Service) cancelExitOrders(ctx context.Context, pos *models.ActivePosition) {
	if pos.StopLossOrder != nil && !pos.StopLossOrder.Filled && pos.StopLossOrder.OrderID != "" {
		if pos.StopLossOrder.IsStop {
			if err := s.bybit.CancelStopOrder(ctx, pos.Symbol, pos.StopLossOrder.OrderID); err != nil {
				s.logger.Warn("cancel sl failed", "symbol", pos.Symbol, "order_id", pos.StopLossOrder.OrderID, "error", err)
			}
		} else {
			if err := s.bybit.CancelOrder(ctx, pos.Symbol, pos.StopLossOrder.OrderID); err != nil {
				s.logger.Warn("cancel sl failed", "symbol", pos.Symbol, "order_id", pos.StopLossOrder.OrderID, "error", err)
			}
		}
	}
	for i := range pos.TakeProfitOrders {
		tp := &pos.TakeProfitOrders[i]
		if !tp.Filled && tp.OrderID != "" {
			if err := s.bybit.CancelOrder(ctx, pos.Symbol, tp.OrderID); err != nil {
				s.logger.Warn("cancel tp failed", "symbol", pos.Symbol, "order_id", tp.OrderID, "kind", tp.Kind, "error", err)
			}
		}
	}
}

func (s *Service) cancelTPOrdersOnly(ctx context.Context, pos *models.ActivePosition) {
	remaining := make([]models.ExitOrder, 0, len(pos.TakeProfitOrders))
	for i := range pos.TakeProfitOrders {
		tp := &pos.TakeProfitOrders[i]
		if tp.Filled {
			remaining = append(remaining, *tp)
			continue
		}
		if tp.OrderID != "" {
			_ = s.bybit.CancelOrder(ctx, pos.Symbol, tp.OrderID)
		}
	}
	pos.TakeProfitOrders = remaining
}

func (s *Service) monitorExitOrders(ctx context.Context, pos *models.ActivePosition) bool {
	if pos.StopLossOrder != nil {
		s.refreshExitOrder(ctx, pos.Symbol, pos.StopLossOrder)
	}

	breakevenHit := false
	for i := 0; i < len(pos.TakeProfitOrders); i++ {
		if i >= len(pos.TakeProfitOrders) {
			break
		}
		tp := &pos.TakeProfitOrders[i]
		if tp.Filled {
			continue
		}
		s.refreshExitOrder(ctx, pos.Symbol, tp)
		if tp.Filled {
			s.logger.Info("tp filled", "symbol", pos.Symbol, "kind", tp.Kind, "price", tp.Price, "qty", tp.FilledQty)
			if tp.Kind == "breakeven" {
				breakevenHit = true
			}
		}
	}

	exSize, hasPos, _ := s.syncPositionFromExchange(ctx, pos)
	if !hasPos || exSize <= 0 {
		s.cancelExitOrders(ctx, pos)
		reason := "exchange_flat"
		if pos.PartialTaken {
			reason = "take_profit_grid"
		}
		s.tryFinalizePosition(ctx, pos, reason, 0)
		return true
	}

	s.syncSLToFullRemainder(ctx, pos)

	if pos.StopLossOrder != nil && pos.StopLossOrder.Filled {
		exSize, hasPos, _ = s.syncPositionFromExchange(ctx, pos)
		if hasPos && hasRemainingSize(exSize, pos.MinOrderQty) {
			pos.StopLossOrder.Filled = false
			pos.StopLossOrder.FilledQty = 0
			s.syncSLToFullRemainder(ctx, pos)
			return false
		}
		s.cancelTPOrdersOnly(ctx, pos)
		closeReason := pos.StopLossOrder.Kind
		if closeReason == "" {
			closeReason = "stop_loss"
		}
		s.tryFinalizePosition(ctx, pos, closeReason, pos.StopLossOrder.FilledPx)
		return true
	}

	if breakevenHit && !pos.BreakevenSet {
		pos.BreakevenSet = true
		pos.PartialTaken = true
		beSL := grid.FeeAwareBreakevenPrice(pos.FillPrice, pos.Direction, s.cfg.FeeBreakevenPct, pos.TickSize)
		slQty := s.slCoverQty(pos, exSize)
		if slQty > 0 {
			if err := s.atomicReplaceStopLoss(ctx, pos, beSL, slQty, "stop_loss"); err != nil {
				s.logger.Warn("move sl to fee-aware breakeven failed", "symbol", pos.Symbol, "error", err)
			}
		}
	}

	// For positions without TPs (min-lot), activate breakeven+trailing when price covers fees.
	if !pos.BreakevenSet && !pos.TimeStopPlaced && len(pos.TakeProfitOrders) == 0 {
		ob, obErr := s.redis.GetOrderbook(ctx, pos.Symbol)
		if obErr == nil {
			mid := grid.MidPrice(ob)
			if mid > 0 {
				feeDist := pos.FillPrice * s.cfg.FeeBreakevenPct
				if (pos.Direction == "LONG" && mid >= pos.FillPrice+feeDist) ||
					(pos.Direction == "SHORT" && mid <= pos.FillPrice-feeDist) {
					pos.BreakevenSet = true
					pos.PartialTaken = true
				}
			}
		}
	}

	// Trailing SL: after breakeven, trail SL to lock in profits.
	if pos.BreakevenSet && !pos.TimeStopPlaced {
		s.maybeTrailStopLoss(ctx, pos, exSize)
	}

	return false
}

// maybeTrailStopLoss moves SL tighter when price has moved significantly in our favor.
// Logic: after breakeven, if price is > 1.5R from entry, trail SL to 0.8R lock.
func (s *Service) maybeTrailStopLoss(ctx context.Context, pos *models.ActivePosition, exSize float64) {
	if pos.StopLossOrder == nil || pos.StopLossOrder.Filled {
		return
	}

	ob, err := s.redis.GetOrderbook(ctx, pos.Symbol)
	if err != nil {
		return
	}
	mid := grid.MidPrice(ob)
	if mid <= 0 {
		return
	}

	risk := math.Abs(pos.FillPrice - pos.StopLoss)
	if risk <= 0 {
		return
	}

	var newSL float64
	switch pos.Direction {
	case "LONG":
		profitDist := mid - pos.FillPrice
		if profitDist < risk*1.0 {
			return
		}
		// Trail: lock 0.8R of current profit
		newSL = mid - risk*0.8
		if newSL <= pos.StopLoss {
			return
		}
	case "SHORT":
		profitDist := pos.FillPrice - mid
		if profitDist < risk*1.0 {
			return
		}
		newSL = mid + risk*0.8
		if newSL >= pos.StopLoss {
			return
		}
	default:
		return
	}

	newSL = grid.RoundToTick(newSL, pos.TickSize)
	// Do NOT enforce min SL distance here — trailing SL intentionally moves SL closer to price.

	// Re-check price: mid may have moved since initial fetch. Bybit rejects trigger on wrong side.
	ob2, ob2Err := s.redis.GetOrderbook(ctx, pos.Symbol)
	if ob2Err == nil {
		now := grid.MidPrice(ob2)
		if now > 0 {
			if pos.Direction == "SHORT" && newSL <= now {
				return
			}
			if pos.Direction == "LONG" && newSL >= now {
				return
			}
		}
	}

	slQty := s.slCoverQty(pos, exSize)
	if slQty <= 0 {
		return
	}

	if err := s.atomicReplaceStopLoss(ctx, pos, newSL, slQty, "trailing_stop"); err != nil {
		s.logger.Warn("trailing SL failed", "symbol", pos.Symbol, "error", err)
		return
	}
	s.logger.Info("trailing SL moved",
		"symbol", pos.Symbol,
		"old_sl", pos.StopLoss,
		"new_sl", newSL,
		"mid", mid,
		"fill", pos.FillPrice,
	)
}

func (s *Service) tryFinalizePosition(ctx context.Context, pos *models.ActivePosition, reason string, fallbackExit float64) {
	if err := s.ensureExchangeFlat(ctx, pos, reason); err != nil {
		s.logger.Error("ensure flat before finalize failed",
			"symbol", pos.Symbol,
			"reason", reason,
			"error", err,
		)
		return
	}
	s.finalizeFromExchange(ctx, pos, reason, fallbackExit)
}

func (s *Service) retryMissingTakeProfits(ctx context.Context, pos *models.ActivePosition, ob models.OrderbookSnapshot) {
	if len(pos.TakeProfitOrders) > 0 {
		return
	}
	if time.Now().UnixMilli()-pos.FilledAt > 60_000 {
		return
	}
	exSize, hasPos, err := s.syncPositionFromExchange(ctx, pos)
	if err != nil || !hasPos || exSize <= 0 {
		return
	}
	if exSize <= pos.MinOrderQty*1.01 {
		pos.ExitGridReady = true
		return
	}
	pos.ExitGridReady = false
	if err := s.deployExitGrid(ctx, pos, ob, pos.PlannedEntry, pos.PlannedSL, pos.TickSize); err != nil {
		s.logger.Warn("exit grid redeploy failed", "symbol", pos.Symbol, "error", err)
	}
}

// timeStopLimitExit is an infrastructure failsafe (GC) when a position outlives TIME_STOP_SECONDS.
func (s *Service) timeStopLimitExit(ctx context.Context, pos *models.ActivePosition, ob models.OrderbookSnapshot) {
	exSize, hasPos, _ := s.syncPositionFromExchange(ctx, pos)
	if !hasPos || exSize <= 0 {
		s.tryFinalizePosition(ctx, pos, "time_stop", 0)
		return
	}

	if pos.TimeStopPlaced {
		if pos.StopLossOrder != nil {
			s.refreshExitOrder(ctx, pos.Symbol, pos.StopLossOrder)
			if pos.StopLossOrder.Filled {
				s.tryFinalizePosition(ctx, pos, "time_stop", pos.StopLossOrder.FilledPx)
			}
		}
		return
	}

	s.cancelTPOrdersOnly(ctx, pos)
	var price float64
	if pos.Direction == "LONG" {
		price = liquidity.BestBid(ob)
	} else {
		price = liquidity.BestAsk(ob)
	}
	if price <= 0 {
		price = grid.MidPrice(ob)
	}
	if price <= 0 {
		s.logger.Warn("time stop skipped, no book price", "symbol", pos.Symbol)
		return
	}

	if err := s.atomicReplaceStopLoss(ctx, pos, price, exSize, "time_stop"); err != nil {
		s.logger.Error("time stop sl replace failed", "symbol", pos.Symbol, "error", err)
		return
	}
	pos.TimeStopPlaced = true
	s.logger.Info("time stop sl placed", "symbol", pos.Symbol, "price", price, "qty", exSize)
}

func (s *Service) syncRemainingSize(ctx context.Context, pos *models.ActivePosition) (float64, bool) {
	size, has, err := s.syncPositionFromExchange(ctx, pos)
	if err != nil {
		return pos.RemainingQty, true
	}
	return size, has
}

func (s *Service) finalizeFromExchange(ctx context.Context, pos *models.ActivePosition, reason string, fallbackExit float64) {
	exSize, hasPos, _ := s.syncPositionFromExchange(ctx, pos)
	if hasPos && exSize > 0 {
		s.logger.Warn("finalize blocked: exchange still open",
			"symbol", pos.Symbol,
			"size", exSize,
			"reason", reason,
		)
		if err := s.ensureExchangeFlat(ctx, pos, reason+"_forced"); err != nil {
			s.logger.Error("forced flatten failed", "symbol", pos.Symbol, "error", err)
			return
		}
	}

	closed, err := s.bybit.WaitForClosedPnL(ctx, pos.Symbol, pos.EntryTime, 8)
	var pnl, entryPrice, exitPrice float64
	exchangePnL := false
	if err == nil && closed != nil && closed.UpdatedTime >= pos.EntryTime {
		pnl = closed.ClosedPnL
		entryPrice = closed.AvgEntryPrice
		exitPrice = closed.AvgExitPrice
		exchangePnL = true
	} else {
		exitPrice = fallbackExit
		if exitPrice <= 0 {
			exitPrice = pos.FillPrice
		}
		entryPrice = pos.FillPrice
		pnl = s.calcPnL(pos, exitPrice)
	}
	s.finalizeClose(ctx, pos, pnl, entryPrice, exitPrice, s.resolveCloseReason(reason, pnl), exchangePnL)
}

// resolveCloseReason maps structural exit triggers to ML-safe labels using actual PnL.
func (s *Service) resolveCloseReason(proposed string, pnl float64) string {
	switch proposed {
	case "stop_loss", "signal_exit", "confidence_decay_exit":
		return proposed
	case "time_stop":
		if pnl > 0 {
			return "take_profit"
		}
		if pnl < 0 {
			return "fee_loss"
		}
		return "exchange_flat"
	}

	if pnl > 0 {
		if proposed == "take_profit_grid" {
			return "take_profit_grid"
		}
		return "take_profit"
	}
	if pnl < 0 {
		return "fee_loss"
	}
	return "exchange_flat"
}

func (s *Service) calcPnL(pos *models.ActivePosition, exit float64) float64 {
	qty := pos.InitialQty
	if pos.RemainingQty > 0 {
		qty = pos.RemainingQty
	}
	if pos.Direction == "LONG" {
		return (exit - pos.FillPrice) * qty
	}
	return (pos.FillPrice - exit) * qty
}

func elapsedMs(since int64) int64 {
	return time.Now().UnixMilli() - since
}

func abs(v float64) float64 {
	if v < 0 {
		return -v
	}
	return v
}
