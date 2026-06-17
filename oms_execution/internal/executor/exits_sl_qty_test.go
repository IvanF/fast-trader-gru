package executor

import (
	"testing"

	"github.com/fast-trader-gru/oms_execution/internal/models"
)

func TestSlCoverQtySubtractsOpenTPs(t *testing.T) {
	s := &Service{}
	pos := &models.ActivePosition{
		QtyStep:     0.01,
		MinOrderQty: 0.01,
		TakeProfitOrders: []models.ExitOrder{
			{Qty: 0.04, Kind: "r_multiple"},
			{Qty: 0.03, Kind: "wall", Filled: true},
		},
	}
	if got := s.slCoverQty(pos, 0.48); got != 0.44 {
		t.Fatalf("slCoverQty = %v, want 0.44", got)
	}
}

func TestSlCoverQtyZeroWhenTPsCoverPosition(t *testing.T) {
	s := &Service{}
	pos := &models.ActivePosition{
		QtyStep:     0.01,
		MinOrderQty: 0.01,
		TakeProfitOrders: []models.ExitOrder{
			{Qty: 0.48, Kind: "r_multiple"},
		},
	}
	if got := s.slCoverQty(pos, 0.48); got != 0 {
		t.Fatalf("slCoverQty = %v, want 0", got)
	}
}
