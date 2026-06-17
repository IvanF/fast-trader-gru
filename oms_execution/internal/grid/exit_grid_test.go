package grid

import (
	"testing"

	"github.com/fast-trader-gru/oms_execution/internal/bybit"
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
	if bybit.NormalizeQty(tpSum+grid.StopLoss.Qty, step, minQty) != bybit.NormalizeQty(total, step, minQty) {
		t.Fatalf("tp+sl = %.4f + %.4f != total %.4f", tpSum, grid.StopLoss.Qty, total)
	}
}
