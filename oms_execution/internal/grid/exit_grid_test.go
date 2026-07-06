package grid

import (
	"testing"
)

func TestTPSum(t *testing.T) {
	grid := ExitGrid{
		StopLoss: ExitLevel{Price: 1.0, Qty: 10.0, Kind: "stop_loss"},
		TakeProfits: []ExitLevel{
			{Price: 1.1, Qty: 5.0, Kind: "fee_aware_tp"},
		},
	}
	sum := TPSum(grid)
	if sum != 5.0 {
		t.Fatalf("expected 5.0, got %.4f", sum)
	}
}
