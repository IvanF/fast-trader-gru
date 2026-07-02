package grid

import (
	"math"
	"testing"
)

func TestFeeAwareBreakevenLong(t *testing.T) {
	price := FeeAwareBreakevenPrice(100.0, "LONG", 0.0015, 0.01, 0.00055, 0.0002)
	// Correct formula: 100 * (1+0.00055) / (1-0.0002) = 100.0770
	// Old (symmetric): 100 * 1.0015 = 100.15
	if price < 100.07-1e-9 {
		t.Fatalf("fee-aware breakeven LONG: got %.4f want >= 100.07", price)
	}
}

func TestFeeAwareBreakevenShort(t *testing.T) {
	price := FeeAwareBreakevenPrice(100.0, "SHORT", 0.0015, 0.01, 0.00055, 0.0002)
	// Correct formula: 100 * (1-0.00055) / (1+0.0002) = 99.9250
	// Old (symmetric): 100 * 0.9985 = 99.85
	if price > 99.93+1e-9 {
		t.Fatalf("fee-aware breakeven SHORT: got %.4f want <= 99.93", price)
	}
}

func TestEnforceMinTPDistanceLong(t *testing.T) {
	tp := EnforceMinTPDistance(100.0, 100.05, "LONG", 0.002, 0.01)
	if tp < 100.2-1e-9 {
		t.Fatalf("TP too close for LONG: got %.4f", tp)
	}
}

func TestFilterTPLevelsByMaxDistanceShort(t *testing.T) {
	levels := []ExitLevel{
		{Price: 0.8018, Kind: "breakeven"},
		{Price: 0.7982, Kind: "liquidity_tp"},
	}
	out := FilterTPLevelsByMaxDistance("SHORT", 0.805, levels, 0.008, 0.0001)
	if len(out) != 1 {
		t.Fatalf("expected 1 level within cap, got %d", len(out))
	}
	if math.Abs(out[0].Price-0.8018) > 1e-6 {
		t.Fatalf("kept level price = %v, want 0.8018", out[0].Price)
	}
}

func TestApplyTPPriceFloorsBreakeven(t *testing.T) {
	levels := []ExitLevel{{Price: 100.01, Qty: 1, Kind: "breakeven"}}
	out := applyTPPriceFloors("LONG", 100.0, levels, 0.002, 0.0015, 0.01, 0.00055, 0.0002)
	// Correct breakeven: 100 * (1+0.00055) / (1-0.0002) = 100.0770
	if out[0].Price < 100.07-1e-9 {
		t.Fatalf("breakeven floor not applied: got %.4f", out[0].Price)
	}
}
