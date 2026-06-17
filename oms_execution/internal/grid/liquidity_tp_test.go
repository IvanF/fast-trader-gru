package grid

import (
	"testing"

	"github.com/fast-trader-gru/oms_execution/internal/models"
)

func TestBuildExitGridHybridNearTPFirst(t *testing.T) {
	signal := models.TradeSignal{
		Direction:             "SHORT",
		Regime:                "Trending",
		VolatilityMultiplier:  0.5,
		TPPrices:              []float64{0.7982, 0.7983},
	}
	opts := ExitGridOptions{
		TPBudgetPct:     0.45,
		MinTPPct:        0.004,
		MaxTPPct:        0.008,
		FeeBreakevenPct: 0.002,
	}
	grid := BuildExitGrid(
		"SHORT",
		0.805,
		0.805,
		0.809,
		models.OrderbookSnapshot{},
		signal,
		0.0001,
		60,
		10,
		10,
		opts,
	)
	if len(grid.TakeProfits) == 0 {
		t.Fatal("expected tp levels")
	}
	firstDist := TPDistancePct(0.805, grid.TakeProfits[0].Price)
	if firstDist > 0.008 {
		t.Fatalf("first tp too far: price=%v dist=%.4f", grid.TakeProfits[0].Price, firstDist)
	}
	if firstDist < 0.003 {
		t.Fatalf("first tp too close: price=%v dist=%.4f", grid.TakeProfits[0].Price, firstDist)
	}
	for _, tp := range grid.TakeProfits {
		if TPDistancePct(0.805, tp.Price) > 0.008+1e-9 {
			t.Fatalf("tp beyond max cap: price=%v kind=%s", tp.Price, tp.Kind)
		}
	}
}

func TestBuildExitGridUsesLiquidityTPPricesWithinCap(t *testing.T) {
	signal := models.TradeSignal{
		Direction: "SHORT",
		TPPrices:  []float64{0.5252, 0.5242, 0.5232},
	}
	opts := ExitGridOptions{
		TPBudgetPct:     0.45,
		MinTPPct:        0.002,
		MaxTPPct:        0.008,
		FeeBreakevenPct: 0.0015,
	}
	grid := BuildExitGrid(
		"SHORT",
		0.5270,
		0.5270,
		0.5320,
		models.OrderbookSnapshot{},
		signal,
		0.0001,
		100,
		1,
		1,
		opts,
	)
	if len(grid.TakeProfits) == 0 {
		t.Fatal("expected liquidity tp levels")
	}
	firstDist := TPDistancePct(0.5270, grid.TakeProfits[0].Price)
	if firstDist > 0.008 {
		t.Fatalf("first tp too far: price=%v dist=%.4f", grid.TakeProfits[0].Price, firstDist)
	}
}
