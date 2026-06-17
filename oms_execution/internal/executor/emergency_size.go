package executor

import (
	"context"

	"github.com/fast-trader-gru/oms_execution/internal/bybit"
	"github.com/fast-trader-gru/oms_execution/internal/models"
)

const sizeViolationMultiplier = 1.5

func positionSizeViolation(currentQty, targetQty float64) bool {
	if targetQty <= 0 || currentQty <= 0 {
		return false
	}
	return currentQty > targetQty*sizeViolationMultiplier
}

// checkPositionSizeViolation trims excess size or flattens when qty exceeds 1.5× target.
func (s *Service) checkPositionSizeViolation(ctx context.Context, pos *models.ActivePosition, exSize float64) bool {
	if pos.EmergencySizeHandled || !positionSizeViolation(exSize, pos.TargetQty) {
		return false
	}
	s.emergencyFlattenSizeViolation(ctx, pos, exSize)
	return true
}

func (s *Service) emergencyFlattenSizeViolation(ctx context.Context, pos *models.ActivePosition, exSize float64) {
	pos.EmergencySizeHandled = true

	s.logger.Error("EMERGENCY_FLATTEN_SIZE_VIOLATION",
		"symbol", pos.Symbol,
		"current_qty", exSize,
		"target_qty", pos.TargetQty,
		"direction", pos.Direction,
	)

	s.cancelEntryOrdersForSymbol(ctx, pos.Symbol)
	s.cancelExitOrders(ctx, pos)

	excess := exSize - pos.TargetQty
	if excess <= pos.MinOrderQty*0.99 {
		return
	}
	excess = bybit.NormalizeQty(excess, pos.QtyStep, pos.MinOrderQty)
	if excess <= 0 {
		return
	}

	side := closeSide(pos.Direction)
	s.logger.Warn("emergency trim excess position",
		"symbol", pos.Symbol,
		"excess_qty", excess,
		"target_qty", pos.TargetQty,
	)
	if _, err := s.bybit.PlaceReduceMarketRetry(ctx, pos.Symbol, side, excess, pos.QtyStep); err != nil {
		s.logger.Error("emergency trim failed, flattening all",
			"symbol", pos.Symbol,
			"error", err,
		)
		if err := s.ensureExchangeFlat(ctx, pos, "EMERGENCY_FLATTEN_SIZE_VIOLATION"); err != nil {
			s.logger.Error("emergency full flatten failed", "symbol", pos.Symbol, "error", err)
		}
		pos.ExitGridReady = false
		return
	}

	newSize, hasPos, err := s.syncPositionFromExchange(ctx, pos)
	if err != nil {
		return
	}
	if hasPos && positionSizeViolation(newSize, pos.TargetQty) {
		s.logger.Error("emergency trim insufficient, flattening all",
			"symbol", pos.Symbol,
			"remaining_qty", newSize,
			"target_qty", pos.TargetQty,
		)
		if err := s.ensureExchangeFlat(ctx, pos, "EMERGENCY_FLATTEN_SIZE_VIOLATION"); err != nil {
			s.logger.Error("emergency full flatten failed", "symbol", pos.Symbol, "error", err)
		}
		pos.ExitGridReady = false
		return
	}

	pos.ExitGridReady = false
	pos.RemainingQty = newSize
	pos.InitialQty = newSize
}

func (s *Service) cancelEntryOrdersForSymbol(ctx context.Context, symbol string) {
	s.mu.Lock()
	p, ok := s.pending[symbol]
	s.mu.Unlock()
	if !ok {
		return
	}
	s.cancelPendingEntry(ctx, p, "emergency_size_violation")
}
