package grid

import (
	"testing"
)

func TestFinalizeGridAllocationReservesSL(t *testing.T) {
	grid := ExitGrid{
		StopLoss: ExitLevel{Price: 1.0, Kind: "stop_loss"},
		TakeProfits: []ExitLevel{
			{Price: 1.1, Qty: 3.0, Kind: "breakeven"},
			{Price: 1.2, Qty: 4.0, Kind: "wall"},
			{Price: 1.3, Qty: 3.0, Kind: "r_multiple"},
			{Price: 1.4, Qty: 2.6, Kind: "trend"},
		},
	}
	total := 12.6
	step := 0.1
	minQty := 0.1

	finalizeGridAllocation(&grid, total, step, minQty, 0.35)

	tpSum := TPSum(grid)
	if tpSum >= total {
		t.Fatalf("tp sum %.4f must be < total %.4f", tpSum, total)
	}
	if grid.StopLoss.Qty <= 0 {
		t.Fatalf("sl qty must be > 0, got %.4f", grid.StopLoss.Qty)
	}
	// SL covers full position (TP orders are reduce-only)
	if grid.StopLoss.Qty < total {
		t.Fatalf("sl qty %.4f must be >= total %.4f (full coverage)", grid.StopLoss.Qty, total)
	}
}
