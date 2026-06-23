package executor

import (
	"context"
	"time"

	"github.com/fast-trader-gru/oms_execution/internal/bybit"
	"github.com/fast-trader-gru/oms_execution/internal/grid"
	"github.com/fast-trader-gru/oms_execution/internal/metrics"
	"github.com/fast-trader-gru/oms_execution/internal/models"
)

func (s *Service) scanOrphanPositions(ctx context.Context) {
	seen := make(map[string]bool)

	positions, err := s.bybit.ListOpenPositions(ctx)
	if err != nil {
		s.logger.Warn("orphan scan list failed", "error", err)
	} else {
		for _, exPos := range positions {
			seen[exPos.Symbol] = true
			s.tryAdoptOrphan(ctx, exPos)
		}
	}

	symbols, err := s.redis.GetActiveSymbols(ctx)
	if err != nil {
		s.logger.Warn("orphan scan active symbols failed", "error", err)
		return
	}
	for _, sym := range symbols {
		if seen[sym] {
			continue
		}
		exPos, err := s.bybit.GetPosition(ctx, sym)
		if err != nil || exPos.Size <= 0 {
			continue
		}
		s.tryAdoptOrphan(ctx, exPos)
	}
}

func (s *Service) tryAdoptOrphan(ctx context.Context, exPos bybit.PositionInfo) {
	s.mu.Lock()
	_, tracked := s.positions[exPos.Symbol]
	_, pending := s.pending[exPos.Symbol]
	until, inCooldown := s.ghostCooldown[exPos.Symbol]
	s.mu.Unlock()
	if tracked || pending {
		return
	}
	if inCooldown && time.Now().UnixMilli() < until {
		return
	}
	if inCooldown {
		s.mu.Lock()
		delete(s.ghostCooldown, exPos.Symbol)
		s.mu.Unlock()
	}
	s.adoptExchangePosition(ctx, exPos, "orphan_scan")
}

func (s *Service) adoptExchangePosition(ctx context.Context, exPos bybit.PositionInfo, reason string) {
	direction := "LONG"
	switch exPos.Side {
	case "Sell":
		direction = "SHORT"
	case "Buy":
		direction = "LONG"
	default:
		s.logger.Warn("orphan position unknown side, skipping", "symbol", exPos.Symbol, "side", exPos.Side)
		return
	}
	qty := exPos.Size
	fillPrice := exPos.AvgPrice
	if fillPrice <= 0 {
		return
	}

	inst, err := s.bybit.GetInstrument(ctx, exPos.Symbol)
	if err != nil {
		inst = bybit.InstrumentInfo{
			TickSize: 0.0001,
			Lot:      bybit.LotFilters{QtyStep: 0.1, MinOrderQty: 0.1, MaxOrderQty: 1e9},
		}
	}
	qty = bybit.NormalizeQty(qty, inst.Lot.QtyStep, inst.Lot.MinOrderQty)
	if qty <= 0 {
		return
	}

	signal := models.TradeSignal{
		Symbol:               exPos.Symbol,
		Direction:            direction,
		Confidence:           s.cfg.ConfidenceThreshold,
		VolatilityMultiplier: 1.0,
		Regime:               "Choppy",
		Timestamp:            time.Now().UnixMilli(),
	}

	ob, err := s.redis.GetOrderbook(ctx, exPos.Symbol)
	if err != nil {
		ob = models.OrderbookSnapshot{Symbol: exPos.Symbol}
	}

	leverage := s.cfg.GetLeverage(exPos.Symbol)
	timeStop := s.cfg.GetTimeStopSeconds(exPos.Symbol)
	minSLPct := s.cfg.GetMinSLPct(exPos.Symbol)

	plan := grid.BuildPlan(signal, ob, inst.TickSize, qty, timeStop, s.planOpts())
	if plan.StopLoss <= 0 {
		plan.StopLoss = fillPrice * 0.99
		if direction == "SHORT" {
			plan.StopLoss = fillPrice * 1.01
		}
	}
	plan.StopLoss = grid.EnforceMinSLDistance(fillPrice, plan.StopLoss, direction, minSLPct, inst.TickSize)

	notional := qty * fillPrice
	marginUSD := notional / float64(max(leverage, 1))

	pos := &models.ActivePosition{
		Symbol:       exPos.Symbol,
		Direction:    direction,
		FillPrice:    fillPrice,
		PlannedEntry: plan.EntryPrice,
		PlannedSL:    plan.StopLoss,
		InitialQty:   qty,
		TargetQty:    qty,
		RemainingQty: qty,
		StopLoss:     plan.StopLoss,
		EntryTime:    time.Now().UnixMilli(),
		TimeStopSec:  timeStop,
		QtyStep:      inst.Lot.QtyStep,
		MinOrderQty:  inst.Lot.MinOrderQty,
		TickSize:     inst.TickSize,
		MarginUSD:    marginUSD,
		NotionalUSD:  notional,
		Leverage:     leverage,
		Signal:       signal,
		FilledAt:     time.Now().UnixMilli(),
	}

	s.mu.Lock()
	if _, exists := s.positions[exPos.Symbol]; exists {
		s.mu.Unlock()
		return
	}
	s.positions[exPos.Symbol] = pos
	metrics.ActivePositions.Set(float64(len(s.positions)))
	metrics.GridActive.WithLabelValues(exPos.Symbol).Set(1)
	s.mu.Unlock()

	s.logger.Warn("adopted orphan exchange position",
		"symbol", exPos.Symbol,
		"reason", reason,
		"direction", direction,
		"fill_price", fillPrice,
		"qty", qty,
		"planned_sl", plan.StopLoss,
	)

	if err := s.deployExitGrid(ctx, pos, ob, plan.EntryPrice, plan.StopLoss, inst.TickSize); err != nil {
		notional := qty * fillPrice
		if len(pos.TakeProfitOrders) == 0 && notional < 15.0 {
			s.logger.Warn("orphan with no TPs and small notional, flattening",
				"symbol", exPos.Symbol, "qty", qty, "notional", notional)
			s.cancelExitOrders(ctx, pos)
			if flatErr := s.ensureExchangeFlat(ctx, pos, "orphan_remainder_close"); flatErr != nil {
				s.logger.Error("orphan flatten failed", "symbol", exPos.Symbol, "error", flatErr)
			} else {
				s.tryFinalizePosition(ctx, pos, "orphan_remainder_close", 0)
				return
			}
		}
		s.logger.Error("orphan exit grid deploy failed", "symbol", exPos.Symbol, "error", err)
	}
	s.publishPositionOpened(ctx, pos)
}
