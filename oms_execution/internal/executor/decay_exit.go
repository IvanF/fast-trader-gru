package executor

import (
	"context"
	"fmt"

	"github.com/fast-trader-gru/oms_execution/internal/grid"
	"github.com/fast-trader-gru/oms_execution/internal/models"
)

const decayProfitLockBufferPct = 0.001

func (s *Service) confidenceDecayExit(ctx context.Context, pos *models.ActivePosition, ob models.OrderbookSnapshot) error {
	if s.isGridDeploying(pos.Symbol) {
		return nil
	}
	switch pos.Signal.DecayReason {
	case "microstructure_adverse":
		return s.decayMicrostructureAdverse(ctx, pos, ob)
	default:
		return s.decayDirectionFlipExit(ctx, pos, ob)
	}
}

// decayMicrostructureAdverse keeps the TP/SL grid; only trails SL when already in profit.
func (s *Service) decayMicrostructureAdverse(ctx context.Context, pos *models.ActivePosition, ob models.OrderbookSnapshot) error {
	mid := grid.MidPrice(ob)
	if mid <= 0 {
		return nil
	}
	if s.tryTrailDecayStopInProfit(ctx, pos, mid) {
		return nil
	}
	s.logger.Info("microstructure decay — exit grid preserved",
		"symbol", pos.Symbol,
		"direction", pos.Direction,
		"confidence", pos.Signal.Confidence,
		"mid", mid,
		"fill", pos.FillPrice,
		"open_tps", len(pos.TakeProfitOrders),
	)
	return nil
}

func (s *Service) tryTrailDecayStopInProfit(ctx context.Context, pos *models.ActivePosition, mid float64) bool {
	if !pos.ExitGridReady || pos.StopLossOrder == nil || pos.StopLossOrder.Filled {
		return false
	}
	minMove := s.cfg.MinTPPct
	if minMove <= 0 {
		minMove = 0.004
	}

	var tighter float64
	switch pos.Direction {
	case "LONG":
		if mid <= pos.FillPrice*(1+minMove) {
			return false
		}
		tighter = pos.FillPrice * (1 + decayProfitLockBufferPct)
		if tighter <= pos.StopLoss {
			return false
		}
	case "SHORT":
		if mid >= pos.FillPrice*(1-minMove) {
			return false
		}
		tighter = pos.FillPrice * (1 + decayProfitLockBufferPct)
		if tighter >= pos.StopLoss {
			return false
		}
	default:
		return false
	}

	exSize, hasPos := s.syncRemainingSize(ctx, pos)
	if !hasPos || exSize <= 0 {
		return false
	}
	slQty := s.slCoverQty(pos, exSize)
	if slQty <= 0 {
		return false
	}

	s.logger.Info("microstructure decay — trailing SL to lock profit",
		"symbol", pos.Symbol,
		"old_sl", pos.StopLoss,
		"new_sl", tighter,
		"mid", mid,
	)
	if err := s.atomicReplaceStopLoss(ctx, pos, tighter, slQty, "stop_loss"); err != nil {
		s.logger.Warn("decay profit trail failed", "symbol", pos.Symbol, "error", err)
		return false
	}
	return true
}

// decayDirectionFlipExit cancels TPs and posts a PostOnly reduce limit at the spread edge.
func (s *Service) decayDirectionFlipExit(ctx context.Context, pos *models.ActivePosition, ob models.OrderbookSnapshot) error {
	s.cancelTPOrdersOnly(ctx, pos)

	exSize, hasPos, err := s.syncPositionFromExchange(ctx, pos)
	if err != nil || !hasPos || exSize <= 0 {
		return err
	}

	price := grid.PassiveMakerExitPrice(pos.Direction, ob, pos.TickSize, s.cfg.EntryMakerTicks)
	if price <= 0 {
		price = grid.MidPrice(ob)
	}
	if price <= 0 {
		return fmt.Errorf("decay maker exit: no book price for %s", pos.Symbol)
	}

	slQty := s.slCoverQty(pos, exSize)
	if slQty <= 0 {
		return nil
	}

	if err := s.atomicReplaceStopLoss(ctx, pos, price, slQty, "confidence_decay_exit"); err != nil {
		return fmt.Errorf("decay maker exit: %w", err)
	}
	pos.TimeStopPlaced = false

	s.logger.Warn("confidence decay — maker reduce exit, TPs cancelled",
		"symbol", pos.Symbol,
		"direction", pos.Direction,
		"decay_reason", pos.Signal.DecayReason,
		"confidence", pos.Signal.Confidence,
		"price", price,
		"qty", slQty,
	)
	return nil
}

func (s *Service) makerSignalExit(ctx context.Context, pos *models.ActivePosition, ob models.OrderbookSnapshot, kind string) error {
	s.cancelTPOrdersOnly(ctx, pos)

	exSize, hasPos, err := s.syncPositionFromExchange(ctx, pos)
	if err != nil || !hasPos || exSize <= 0 {
		return err
	}

	price := grid.PassiveMakerExitPrice(pos.Direction, ob, pos.TickSize, s.cfg.EntryMakerTicks)
	if price <= 0 {
		price = grid.MidPrice(ob)
	}
	if price <= 0 {
		return fmt.Errorf("%s: no book price for %s", kind, pos.Symbol)
	}

	slQty := s.slCoverQty(pos, exSize)
	if slQty <= 0 {
		return nil
	}

	if err := s.atomicReplaceStopLoss(ctx, pos, price, slQty, kind); err != nil {
		return fmt.Errorf("%s: %w", kind, err)
	}
	pos.TimeStopPlaced = false

	s.logger.Warn("adverse signal — maker reduce exit, TPs cancelled",
		"symbol", pos.Symbol,
		"direction", pos.Direction,
		"kind", kind,
		"price", price,
		"qty", slQty,
	)
	return nil
}
