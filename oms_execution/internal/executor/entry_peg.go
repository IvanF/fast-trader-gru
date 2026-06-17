package executor

import (
	"context"

	"github.com/fast-trader-gru/oms_execution/internal/bybit"
	"github.com/fast-trader-gru/oms_execution/internal/grid"
	"github.com/fast-trader-gru/oms_execution/internal/models"
)

// maybePegPendingEntry reprices a pending limit toward the current maker edge while setup is valid.
func (s *Service) maybePegPendingEntry(ctx context.Context, p *models.PendingEntry, ob models.OrderbookSnapshot) {
	if s.pendingEntryIsCancelling(p) {
		return
	}
	if p.Signal.Confidence < s.cfg.ConfidenceThreshold {
		return
	}

	inst, err := s.bybit.GetInstrument(ctx, p.Symbol)
	if err != nil {
		inst = bybit.InstrumentInfo{TickSize: p.TickSize, Lot: bybit.LotFilters{QtyStep: p.QtyStep, MinOrderQty: p.MinOrderQty}}
	}

	plan := grid.BuildPlan(p.Signal, ob, inst.TickSize, p.Qty, p.TimeStopSec, s.planOpts())
	if p.Signal.StopLoss > 0 {
		plan.StopLoss = p.Signal.StopLoss
	}
	if len(p.Signal.TakeProfits) > 0 {
		plan.TakeProfits = p.Signal.TakeProfits
	}
	if s.cfg.EntryMakerTicks > 0 {
		if makerEntry := grid.AggressiveMakerEntry(plan.Direction, ob, inst.TickSize, s.cfg.EntryMakerTicks); makerEntry > 0 {
			plan.EntryPrice = makerEntry
		}
	} else if p.Signal.EntryPrice > 0 {
		plan.EntryPrice = p.Signal.EntryPrice
	}

	if priceDriftPct(p.EntryPrice, plan.EntryPrice) < s.cfg.EntryRepriceThresholdPct {
		return
	}

	_ = s.replacePendingEntry(ctx, p, p.Signal, plan, inst, ob, "peg_reprice")
}
