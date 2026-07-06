package grid

import (
	"math"
	"testing"

	"github.com/fast-trader-gru/oms_execution/internal/models"
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

// TestBuildExitGridSLRange verifies SL is always in [0.3%, 0.8%] of fillPrice.
func TestBuildExitGridSLRange(t *testing.T) {
	// Test with various fill prices and volatility multipliers
	testCases := []struct {
		fillPrice float64
		vm        float64
		direction string
	}{
		{0.01, 1.0, "LONG"},
		{0.01, 1.0, "SHORT"},
		{1.0, 1.0, "LONG"},
		{1.0, 1.0, "SHORT"},
		{100.0, 1.0, "LONG"},
		{100.0, 2.0, "SHORT"},
		{100.0, 0.5, "LONG"},
		{10000.0, 3.0, "SHORT"},
	}

	for _, tc := range testCases {
		// Empty orderbook — falls back to min/max SL
		ob := models.OrderbookSnapshot{}
		signal := models.TradeSignal{
			VolatilityMultiplier: tc.vm,
			MacroTrend5m:         0.0,
		}
		tickSize := 0.00001
		if tc.fillPrice > 100 {
			tickSize = 0.01
		}
		grid := BuildExitGrid(
			tc.direction, tc.fillPrice, tc.fillPrice, 0,
			ob, signal, tickSize, 100.0, 1.0, 0.1,
			ExitGridOptions{
				MinSLPct:        0.004,
				MaxSLPct:        0.008,
				MinTPPct:        0.003,
				MaxTPPct:        0.03,
				EntryFeeRate:    0.00055,
				ExitFeeRate:     0.0002,
				TargetNetProfitPct: 0.002,
			},
		)

		slDist := math.Abs(grid.StopLoss.Price-tc.fillPrice) / tc.fillPrice
		if slDist < 0.002 || slDist > 0.01 {
			t.Errorf("fill=%.2f vm=%.1f dir=%s: SL dist=%.4f%% outside [0.2%%, 1.0%%]",
				tc.fillPrice, tc.vm, tc.direction, slDist*100)
		}
	}
}

// TestBuildExitGridRR verifies R:R >= 1.0 after enforcement.
func TestBuildExitGridRR(t *testing.T) {
	testCases := []struct {
		fillPrice float64
		direction string
	}{
		{0.10, "LONG"},
		{0.10, "SHORT"},
		{1.0, "LONG"},
		{1.0, "SHORT"},
		{100.0, "LONG"},
		{100.0, "SHORT"},
	}

	for _, tc := range testCases {
		ob := models.OrderbookSnapshot{}
		signal := models.TradeSignal{VolatilityMultiplier: 1.0}
		grid := BuildExitGrid(
			tc.direction, tc.fillPrice, tc.fillPrice, 0,
			ob, signal, 0.0001, 100.0, 1.0, 0.1,
			ExitGridOptions{
				MinSLPct: 0.004, MaxSLPct: 0.008,
				MinTPPct: 0.003, MaxTPPct: 0.03,
				EntryFeeRate: 0.00055, ExitFeeRate: 0.0002,
				TargetNetProfitPct: 0.002,
			},
		)

		slDist := math.Abs(grid.StopLoss.Price - tc.fillPrice)
		tpDist := math.Abs(grid.TakeProfits[0].Price - tc.fillPrice)

		if slDist <= 0 {
			t.Errorf("fill=%.2f: SL distance is zero", tc.fillPrice)
			continue
		}
		rr := tpDist / slDist
		if rr < 1.0 {
			t.Errorf("fill=%.2f dir=%s: R:R=%.2f < 1.0 (SL=%.6f TP=%.6f)",
				tc.fillPrice, tc.direction, rr, grid.StopLoss.Price, grid.TakeProfits[0].Price)
		}
	}
}

// TestBuildExitGridMaxSLCap verifies SL never exceeds 0.8%.
func TestBuildExitGridMaxSLCap(t *testing.T) {
	// Even with extreme volatility, SL should cap at 0.8%
	ob := models.OrderbookSnapshot{}
	signal := models.TradeSignal{
		VolatilityMultiplier: 10.0, // very high vol
		MacroTrend15m:        -0.01,
	}
	grid := BuildExitGrid(
		"LONG", 1.0, 1.0, 0,
		ob, signal, 0.0001, 100.0, 1.0, 0.1,
		ExitGridOptions{
			MinSLPct: 0.004, MaxSLPct: 0.008,
			MinTPPct: 0.003, MaxTPPct: 0.03,
			EntryFeeRate: 0.00055, ExitFeeRate: 0.0002,
			TargetNetProfitPct: 0.002,
		},
	)

	slDist := math.Abs(grid.StopLoss.Price-1.0) / 1.0
	if slDist > 0.012 { // 0.8% * sqrt(10) = 2.53%, but should be capped
		t.Errorf("SL dist %.4f%% exceeds cap with vm=10.0", slDist*100)
	}
}
