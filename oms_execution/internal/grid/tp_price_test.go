package grid

import (
	"math"
	"testing"
)

func TestCalculateExitPrice(t *testing.T) {
	tests := []struct {
		name            string
		entryPrice      float64
		direction       string
		entryFeeRate    float64
		exitFeeRate     float64
		targetNetProfit float64
		tickSize        float64
	}{
		{
			name:            "BTC LONG 2% target, taker+maker fees",
			entryPrice:      60000,
			direction:       "LONG",
			entryFeeRate:    0.00055,
			exitFeeRate:     0.0002,
			targetNetProfit: 0.02,
			tickSize:        0.1,
		},
		{
			name:            "ETH SHORT 1% target",
			entryPrice:      3500,
			direction:       "SHORT",
			entryFeeRate:    0.00055,
			exitFeeRate:     0.0002,
			targetNetProfit: 0.01,
			tickSize:        0.01,
		},
		{
			name:            "XRP LONG 0.5% target",
			entryPrice:      2.2,
			direction:       "LONG",
			entryFeeRate:    0.00055,
			exitFeeRate:     0.0002,
			targetNetProfit: 0.005,
			tickSize:        0.0001,
		},
		{
			name:            "XRP SHORT 0.5% target",
			entryPrice:      2.2,
			direction:       "SHORT",
			entryFeeRate:    0.00055,
			exitFeeRate:     0.0002,
			targetNetProfit: 0.005,
			tickSize:        0.0001,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			price := CalculateExitPrice(
				tt.entryPrice, tt.direction,
				tt.entryFeeRate, tt.exitFeeRate,
				tt.targetNetProfit, tt.tickSize,
			)

			if price <= 0 {
				t.Fatalf("expected positive exit price, got %f", price)
			}

			if tt.direction == "LONG" && price <= tt.entryPrice {
				t.Errorf("LONG exit %f must be > entry %f", price, tt.entryPrice)
			}
			if tt.direction == "SHORT" && price >= tt.entryPrice {
				t.Errorf("SHORT exit %f must be < entry %f", price, tt.entryPrice)
			}

			rem := math.Remainder(price, tt.tickSize)
			if math.Abs(rem) > 1e-10 {
				t.Errorf("price %f not aligned to tick %f (remainder %f)", price, tt.tickSize, rem)
			}

			t.Logf("%s entry=%.4f exit=%.4f dist=%.4f%%",
				tt.direction, tt.entryPrice, price,
				math.Abs(price-tt.entryPrice)/tt.entryPrice*100)
		})
	}
}

func TestCalculateExitPrice_Rounding(t *testing.T) {
	// LONG: raw = 100 * (1 + 0.00055 + 0.02) / (1 - 0.0002) ≈ 102.0756
	// Must round UP to 102.08
	price := CalculateExitPrice(100.0, "LONG", 0.00055, 0.0002, 0.02, 0.01)
	if price < 102.08 {
		t.Errorf("LONG expected >= 102.08, got %f", price)
	}
	t.Logf("LONG: raw≈102.0756 → rounded to %.2f", price)

	// SHORT: raw = 100 * (1 - 0.00055 - 0.02) / (1 + 0.0002) ≈ 97.9256
	// Must round DOWN to 97.92
	price = CalculateExitPrice(100.0, "SHORT", 0.00055, 0.0002, 0.02, 0.01)
	if price > 97.92 {
		t.Errorf("SHORT expected <= 97.92, got %f", price)
	}
	t.Logf("SHORT: raw≈97.9256 → rounded to %.2f", price)
}

func TestCalculateExitPrice_Symmetry(t *testing.T) {
	entry := 100.0
	tick := 0.01
	fee := 0.00055
	makerFee := 0.0002
	target := 0.02

	longPrice := CalculateExitPrice(entry, "LONG", fee, makerFee, target, tick)
	shortPrice := CalculateExitPrice(entry, "SHORT", fee, makerFee, target, tick)

	longDist := (longPrice - entry) / entry
	shortDist := (entry - shortPrice) / entry

	t.Logf("LONG  exit=%.4f dist=%.4f%%", longPrice, longDist*100)
	t.Logf("SHORT exit=%.4f dist=%.4f%%", shortPrice, shortDist*100)

	diff := math.Abs(longDist - shortDist)
	if diff > 0.001 {
		t.Errorf("asymmetry too large: diff=%.6f", diff)
	}
}
