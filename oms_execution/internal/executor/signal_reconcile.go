package executor

import (
	"context"
	"math"
	"time"

	"github.com/fast-trader-gru/oms_execution/internal/bybit"
	"github.com/fast-trader-gru/oms_execution/internal/grid"
	"github.com/fast-trader-gru/oms_execution/internal/models"
)

func (s *Service) reconcileExistingSignal(ctx context.Context, signal models.TradeSignal, recvAt time.Time) error {
	signal = s.capSignalVol(signal)
	s.mu.Lock()
	if until, ok := s.ghostCooldown[signal.Symbol]; ok {
		s.mu.Unlock()
		if time.Now().UnixMilli() < until {
			return nil
		}
		s.mu.Lock()
		delete(s.ghostCooldown, signal.Symbol)
	}
	if pos, ok := s.positions[signal.Symbol]; ok {
		s.mu.Unlock()
		return s.reconcileActivePosition(ctx, signal, pos)
	}
	if pending, ok := s.pending[signal.Symbol]; ok {
		s.mu.Unlock()
		return s.reconcilePendingEntry(ctx, signal, pending, recvAt)
	}
	s.mu.Unlock()
	return s.placeNewEntry(ctx, signal, recvAt)
}

func (s *Service) reconcilePendingEntry(ctx context.Context, signal models.TradeSignal, p *models.PendingEntry, recvAt time.Time) error {
	signal = s.capSignalVol(signal)
	if s.pendingEntryIsCancelling(p) {
		return nil
	}
	if signal.SetupAction == "abort_setup" {
		reason := "abort_setup"
		if signal.AbortReason != "" {
			reason = signal.AbortReason
		}
		s.cancelPendingEntry(ctx, p, reason)
		return nil
	}
	if signal.Direction != p.Direction {
		s.cancelPendingEntry(ctx, p, "signal_direction_flip")
		return nil
	}
	if signal.Confidence < s.cfg.ConfidenceThreshold {
		s.cancelPendingEntry(ctx, p, "signal_confidence_low")
		return nil
	}

	ob, err := s.redis.GetOrderbook(ctx, signal.Symbol)
	if err != nil {
		return err
	}
	mid := grid.MidPrice(ob)
	if mid <= 0 {
		return nil
	}

	if s.pendingEntryStale(p, mid) {
		s.cancelPendingEntry(ctx, p, "entry_stale_market_moved")
		return nil
	}

	inst, err := s.bybit.GetInstrument(ctx, signal.Symbol)
	if err != nil {
		inst = bybit.InstrumentInfo{TickSize: p.TickSize, Lot: bybit.LotFilters{QtyStep: p.QtyStep, MinOrderQty: p.MinOrderQty}}
	}

	plan := grid.BuildPlan(signal, ob, inst.TickSize, p.Qty, p.TimeStopSec, s.planOpts())
	if signal.StopLoss > 0 {
		plan.StopLoss = signal.StopLoss
	}
	if len(signal.TakeProfits) > 0 {
		plan.TakeProfits = signal.TakeProfits
	}
	if s.cfg.EntryMakerTicks > 0 {
		if makerEntry := grid.AggressiveMakerEntry(plan.Direction, ob, inst.TickSize, s.cfg.EntryMakerTicks); makerEntry > 0 {
			plan.EntryPrice = makerEntry
		}
	} else if signal.EntryPrice > 0 {
		plan.EntryPrice = signal.EntryPrice
	}
	if priceDriftPct(p.EntryPrice, plan.EntryPrice) >= s.cfg.EntryRepriceThresholdPct {
		return s.replacePendingEntry(ctx, p, signal, plan, inst, ob, "signal_reprice")
	}

	if signal.Regime != p.Signal.Regime || math.Abs(signal.VolatilityMultiplier-p.Signal.VolatilityMultiplier) > s.cfg.PendingVolRepriceDelta {
		p.Signal = signal
		p.StopLoss = plan.StopLoss
		p.TakeProfits = plan.TakeProfits
		p.Orderbook = ob
		s.logger.Info("pending entry plan updated (same price)",
			"symbol", p.Symbol,
			"regime", signal.Regime,
			"sl", plan.StopLoss,
		)
	}
	return nil
}

func (s *Service) reconcileActivePosition(ctx context.Context, signal models.TradeSignal, pos *models.ActivePosition) error {
	ob, err := s.redis.GetOrderbook(ctx, signal.Symbol)
	if err != nil {
		return err
	}

	if signal.ExitReason == "confidence_decay" {
		if s.isGridDeploying(signal.Symbol) {
			s.logger.Info("confidence decay deferred — exit grid deploying",
				"symbol", signal.Symbol,
			)
			return nil
		}
		pos.Signal = signal
		return s.confidenceDecayExit(ctx, pos, ob)
	}

	if signal.Direction != pos.Direction {
		return s.handleAdverseSignal(ctx, pos, signal, ob)
	}
	if signal.Confidence < s.cfg.ConfidenceThreshold {
		return s.tightenExitOnWeakSignal(ctx, pos, ob, "signal_confidence_low")
	}

	pos.Signal = signal
	if !pos.ExitGridReady {
		return nil
	}

	plan := grid.BuildPlan(signal, ob, pos.TickSize, pos.RemainingQty, pos.TimeStopSec, s.planOpts())
	if s.exitGridNeedsRefresh(pos, plan, ob) {
		return s.redeployExitGrid(ctx, pos, ob, plan.EntryPrice, pos.StopLoss, "signal_regime_update")
	}
	return nil
}

func (s *Service) handleAdverseSignal(ctx context.Context, pos *models.ActivePosition, signal models.TradeSignal, ob models.OrderbookSnapshot) error {
	pos.Signal = signal
	return s.makerSignalExit(ctx, pos, ob, "signal_exit")
}

func (s *Service) tightenExitOnWeakSignal(ctx context.Context, pos *models.ActivePosition, ob models.OrderbookSnapshot, reason string) error {
	if !pos.ExitGridReady || pos.StopLossOrder == nil || pos.StopLossOrder.Filled {
		return nil
	}

	mid := grid.MidPrice(ob)
	if mid <= 0 {
		return nil
	}

	var tighter float64
	if pos.Direction == "LONG" {
		tighter = mid * (1 - 0.001)
		if pos.PartialTaken {
			tighter = math.Max(tighter, pos.FillPrice)
		}
		if tighter <= pos.StopLoss {
			return nil
		}
	} else {
		tighter = mid * (1 + 0.001)
		if pos.PartialTaken {
			tighter = math.Min(tighter, pos.FillPrice)
		}
		if tighter >= pos.StopLoss {
			return nil
		}
	}

	exSize, hasPos := s.syncRemainingSize(ctx, pos)
	if !hasPos || exSize <= 0 {
		return nil
	}

	s.logger.Info("tightening stop on weak signal",
		"symbol", pos.Symbol,
		"reason", reason,
		"old_sl", pos.StopLoss,
		"new_sl", tighter,
	)
	slQty := s.slCoverQty(pos, exSize)
	if slQty <= 0 {
		return nil
	}
	return s.replaceStopLoss(ctx, pos, tighter, slQty)
}

func (s *Service) maybeRefreshExitGrid(ctx context.Context, pos *models.ActivePosition, ob models.OrderbookSnapshot) {
	if !pos.ExitGridReady || pos.TimeStopPlaced {
		return
	}
	if pos.LastGridDeployAt > 0 {
		elapsed := time.Now().UnixMilli() - pos.LastGridDeployAt
		if elapsed < int64(s.cfg.ExitGridMinRedeploySec)*1000 {
			return
		}
	}

	plan := grid.BuildPlan(pos.Signal, ob, pos.TickSize, pos.RemainingQty, pos.TimeStopSec, s.planOpts())
	if s.exitGridNeedsRefresh(pos, plan, ob) {
		_ = s.redeployExitGrid(ctx, pos, ob, plan.EntryPrice, pos.StopLoss, "market_drift")
	}
}

func (s *Service) redeployExitGrid(
	ctx context.Context,
	pos *models.ActivePosition,
	ob models.OrderbookSnapshot,
	plannedEntry, plannedSL float64,
	reason string,
) error {
	s.setGridDeploying(pos.Symbol, true)
	defer s.setGridDeploying(pos.Symbol, false)

	if pos.LastGridDeployAt > 0 {
		elapsed := time.Now().UnixMilli() - pos.LastGridDeployAt
		if elapsed < int64(s.cfg.ExitGridMinRedeploySec)*1000 {
			s.syncSLToFullRemainder(ctx, pos)
			return nil
		}
	}

	exSize, hasPos, err := s.syncPositionFromExchange(ctx, pos)
	if err != nil || !hasPos || exSize <= 0 {
		return err
	}

	plan := grid.BuildPlan(pos.Signal, ob, pos.TickSize, exSize, pos.TimeStopSec, s.planOpts())
	if plannedSL <= 0 {
		plannedSL = plan.StopLoss
	}
	newSL := s.enforceSLPrice(ctx, pos, plannedSL, pos.TickSize)
	newSL = clampSLTightenOnly(pos.Direction, pos.StopLoss, newSL)

	slQty := s.slCoverQty(pos, exSize)
	if priceDriftPct(pos.StopLoss, newSL) >= s.cfg.ExitRepriceThresholdPct {
		if slQty > 0 && !slWouldWiden(pos.Direction, pos.StopLoss, newSL) {
			if err := s.atomicReplaceStopLoss(ctx, pos, newSL, slQty, "stop_loss"); err != nil {
				return err
			}
		}
		pos.PlannedSL = newSL
	} else {
		s.syncSLToFullRemainder(ctx, pos)
	}

	if s.tpGridNeedsRefresh(pos, plan, ob, exSize) {
		elapsed := time.Now().UnixMilli() - pos.LastGridDeployAt
		if elapsed >= int64(s.cfg.ExitGridMinRedeploySec)*1000 {
		tpRefPrice := pos.FillPrice
		opts := s.exitGridOptsForSymbol(pos.Symbol)
		opts.MaxTPPct = 0

		var exitGrid grid.ExitGrid
		if exSize < pos.MinOrderQty*2 {
			tpPrice := pos.FillPrice
			if pos.Direction == "SHORT" {
				tpPrice = grid.FeeAwareBreakevenPrice(pos.FillPrice, "SHORT", opts.FeeBreakevenPct, pos.TickSize)
			} else {
				tpPrice = grid.FeeAwareBreakevenPrice(pos.FillPrice, "LONG", opts.FeeBreakevenPct, pos.TickSize)
			}
			mid := grid.MidPrice(ob)
			if mid > 0 {
				if pos.Direction == "SHORT" && tpPrice >= mid {
					tpPrice = grid.RoundToTick(mid-pos.FillPrice*opts.MinTPPct, pos.TickSize)
				}
				if pos.Direction == "LONG" && tpPrice <= mid {
					tpPrice = grid.RoundToTick(mid+pos.FillPrice*opts.MinTPPct, pos.TickSize)
				}
			}
			exitGrid = grid.ExitGrid{
				StopLoss: grid.ExitLevel{Price: newSL, Kind: "stop_loss"},
				TakeProfits: []grid.ExitLevel{
					{Price: tpPrice, Qty: exSize, Kind: "breakeven"},
				},
			}
		} else {
			exitGrid = grid.BuildExitGrid(
				pos.Direction, tpRefPrice, plannedEntry, newSL, ob, pos.Signal,
				pos.TickSize, exSize, pos.QtyStep, pos.MinOrderQty, opts,
			)
			mid := grid.MidPrice(ob)
			if mid > 0 {
				filtered := exitGrid.TakeProfits[:0]
				for _, tp := range exitGrid.TakeProfits {
					if pos.Direction == "SHORT" && tp.Price >= mid {
						continue
					}
					if pos.Direction == "LONG" && tp.Price <= mid {
						continue
					}
					filtered = append(filtered, tp)
				}
				exitGrid.TakeProfits = filtered
			}
		}

		if len(exitGrid.TakeProfits) == 0 {
			bePrice := grid.FeeAwareBreakevenPrice(pos.FillPrice, pos.Direction, opts.FeeBreakevenPct, pos.TickSize)
			mid := grid.MidPrice(ob)
			if mid > 0 {
				if pos.Direction == "SHORT" && bePrice >= mid {
					bePrice = grid.RoundToTick(mid-pos.FillPrice*opts.MinTPPct, pos.TickSize)
				}
				if pos.Direction == "LONG" && bePrice <= mid {
					bePrice = grid.RoundToTick(mid+pos.FillPrice*opts.MinTPPct, pos.TickSize)
				}
			}
			tpID, err := s.bybit.PlaceReduceLimit(ctx, pos.Symbol, closeSide(pos.Direction), exSize, pos.QtyStep, bybit.FormatPrice(bePrice))
			if err == nil {
				pos.TakeProfitOrders = []models.ExitOrder{{OrderID: tpID, Price: bePrice, Qty: exSize, Kind: "breakeven"}}
				pos.LastGridDeployAt = time.Now().UnixMilli()
				return nil
			}
			pos.LastGridDeployAt = time.Now().UnixMilli()
			return nil
		}
		s.logger.Info("redeploying tp ladder only",
			"symbol", pos.Symbol,
			"reason", reason,
			"qty", exSize,
			"tp_count", len(exitGrid.TakeProfits),
		)
		s.cancelTPOrdersOnly(ctx, pos)
		side := closeSide(pos.Direction)
		for _, tp := range exitGrid.TakeProfits {
			if tp.Qty <= 0 {
				continue
			}
			tpID, err := s.bybit.PlaceReduceLimit(ctx, pos.Symbol, side, tp.Qty, pos.QtyStep, bybit.FormatPrice(tp.Price))
			if err != nil {
				s.logger.Warn("tp redeploy failed", "symbol", pos.Symbol, "kind", tp.Kind, "error", err)
				continue
			}
			pos.TakeProfitOrders = append(pos.TakeProfitOrders, models.ExitOrder{
				OrderID: tpID, Price: tp.Price, Qty: tp.Qty, Kind: tp.Kind,
			})
		}
		s.syncSLToFullRemainder(ctx, pos)
		}
	}

	pos.LastGridDeployAt = time.Now().UnixMilli()
	return nil
}

func (s *Service) tpGridNeedsRefresh(pos *models.ActivePosition, plan models.GridPlan, ob models.OrderbookSnapshot, exSize float64) bool {
	actualSL := pos.StopLoss
	if actualSL <= 0 {
		actualSL = plan.StopLoss
	}
	fresh := grid.BuildExitGrid(
		pos.Direction, pos.FillPrice, plan.EntryPrice, actualSL, ob, pos.Signal,
		pos.TickSize, exSize, pos.QtyStep, pos.MinOrderQty, s.exitGridOptsForSymbol(pos.Symbol),
	)
	mid := grid.MidPrice(ob)
	if mid > 0 {
		filtered := fresh.TakeProfits[:0]
		for _, tp := range fresh.TakeProfits {
			if pos.Direction == "SHORT" && tp.Price >= mid {
				continue
			}
			if pos.Direction == "LONG" && tp.Price <= mid {
				continue
			}
			filtered = append(filtered, tp)
		}
		fresh.TakeProfits = filtered
	}
	for _, tp := range pos.TakeProfitOrders {
		if tp.Filled {
			continue
		}
		if !tpStillValid(tp, fresh.TakeProfits) {
			return true
		}
	}
	for _, freshTP := range fresh.TakeProfits {
		if !freshTPExists(freshTP, pos.TakeProfitOrders) {
			return true
		}
	}
	return false
}

func (s *Service) refreshAllExitOrders(ctx context.Context, pos *models.ActivePosition) {
	if pos.StopLossOrder != nil {
		s.refreshExitOrder(ctx, pos.Symbol, pos.StopLossOrder)
	}
	for i := range pos.TakeProfitOrders {
		s.refreshExitOrder(ctx, pos.Symbol, &pos.TakeProfitOrders[i])
	}
}

func (s *Service) cancelUnfilledExits(ctx context.Context, pos *models.ActivePosition) {
	if pos.StopLossOrder != nil && !pos.StopLossOrder.Filled && pos.StopLossOrder.OrderID != "" {
		_ = s.bybit.CancelOrder(ctx, pos.Symbol, pos.StopLossOrder.OrderID)
	}
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

func (s *Service) exitGridNeedsRefresh(pos *models.ActivePosition, plan models.GridPlan, ob models.OrderbookSnapshot) bool {
	if pos.StopLossOrder == nil || pos.StopLossOrder.Filled {
		return false
	}
	return s.tpGridNeedsRefresh(pos, plan, ob, pos.RemainingQty)
}

func tpStillValid(existing models.ExitOrder, fresh []grid.ExitLevel) bool {
	for _, f := range fresh {
		if existing.Kind == f.Kind && priceDriftPct(existing.Price, f.Price) < 0.003 {
			return true
		}
	}
	return false
}

func freshTPExists(fresh grid.ExitLevel, existing []models.ExitOrder) bool {
	for _, tp := range existing {
		if tp.Filled {
			continue
		}
		if tp.Kind == fresh.Kind && priceDriftPct(tp.Price, fresh.Price) < 0.003 {
			return true
		}
	}
	return false
}

func (s *Service) pendingEntryStale(p *models.PendingEntry, mid float64) bool {
	if p.EntryPrice <= 0 || mid <= 0 {
		return false
	}
	away := math.Abs(mid-p.EntryPrice) / p.EntryPrice
	if away < s.cfg.StaleEntryMoveAwayPct {
		return false
	}
	if p.Direction == "LONG" && mid > p.EntryPrice*(1+s.cfg.StaleEntryMoveAwayPct) {
		return true
	}
	if p.Direction == "SHORT" && mid < p.EntryPrice*(1-s.cfg.StaleEntryMoveAwayPct) {
		return true
	}
	return false
}

func priceDriftPct(a, b float64) float64 {
	if a <= 0 || b <= 0 {
		return 1
	}
	return math.Abs(a-b) / a
}
