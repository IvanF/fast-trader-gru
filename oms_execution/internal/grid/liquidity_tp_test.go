package grid

import (
	"testing"

	"github.com/fast-trader-gru/oms_execution/internal/models"
)

func TestBuildExitGridSingleTP(t *testing.T) {
	signal := models.TradeSignal{
		Direction:            "SHORT",
		Regime:               "Trending",
		VolatilityMultiplier: 0.5,
	}
	opts := ExitGridOptions{
		TPBudgetPct:        0.45,
		MinTPPct:           0.004,
		MaxTPPct:           0.008,
		FeeBreakevenPct:    0.002,
		EntryFeeRate:       0.00055,
		ExitFeeRate:        0.0002,
		TargetNetProfitPct: 0.002,
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
	if len(grid.TakeProfits) != 1 {
		t.Fatalf("expected exactly 1 TP, got %d", len(grid.TakeProfits))
	}
	tp := grid.TakeProfits[0]
	if tp.Price >= 0.805 {
		t.Fatalf("SHORT TP must be below entry: got %f", tp.Price)
	}
	if tp.Kind != "fee_aware_tp" {
		t.Fatalf("expected kind=fee_aware_tp, got %s", tp.Kind)
	}
	dist := TPDistancePct(0.805, tp.Price)
	t.Logf("SHORT entry=0.805 TP=%f dist=%.4f%% kind=%s qty=%f", tp.Price, dist*100, tp.Kind, tp.Qty)
}

func TestBuildExitGridSingleTP_Long(t *testing.T) {
	signal := models.TradeSignal{
		Direction:            "LONG",
		Regime:               "Choppy",
		VolatilityMultiplier: 1.0,
	}
	opts := ExitGridOptions{
		TPBudgetPct:        0.45,
		MinTPPct:           0.004,
		MaxTPPct:           0.008,
		FeeBreakevenPct:    0.002,
		EntryFeeRate:       0.00055,
		ExitFeeRate:        0.0002,
		TargetNetProfitPct: 0.002,
	}
	grid := BuildExitGrid(
		"LONG",
		2.20,
		2.20,
		2.18,
		models.OrderbookSnapshot{},
		signal,
		0.0001,
		100,
		1,
		1,
		opts,
	)
	if len(grid.TakeProfits) != 1 {
		t.Fatalf("expected exactly 1 TP, got %d", len(grid.TakeProfits))
	}
	tp := grid.TakeProfits[0]
	if tp.Price <= 2.20 {
		t.Fatalf("LONG TP must be above entry: got %f", tp.Price)
	}
	dist := TPDistancePct(2.20, tp.Price)
	t.Logf("LONG entry=2.20 TP=%f dist=%.4f%% kind=%s qty=%f", tp.Price, dist*100, tp.Kind, tp.Qty)
}

func TestBuildExitGridSLAlwaysPresent(t *testing.T) {
	signal := models.TradeSignal{Direction: "SHORT", VolatilityMultiplier: 1.0}
	opts := ExitGridOptions{
		MinSLPct:           0.003,
		MaxSLPct:           0.005,
		EntryFeeRate:       0.00055,
		ExitFeeRate:        0.0002,
		TargetNetProfitPct: 0.002,
	}
	grid := BuildExitGrid(
		"SHORT",
		100.0,
		100.0,
		101.0,
		models.OrderbookSnapshot{},
		signal,
		0.01,
		50,
		1,
		1,
		opts,
	)
	if grid.StopLoss.Price <= 0 {
		t.Fatal("SL must be set")
	}
	if grid.StopLoss.Price <= 100.0 {
		t.Fatalf("SHORT SL must be above entry: got %f", grid.StopLoss.Price)
	}
}
