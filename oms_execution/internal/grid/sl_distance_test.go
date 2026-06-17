package grid

import (
	"testing"
)

func TestEnforceMinSLDistanceShort(t *testing.T) {
	sl := EnforceMinSLDistance(0.52, 0.5192, "SHORT", 0.005, 0.0001)
	if sl < 0.52*1.005-1e-6 {
		t.Fatalf("SL too close for SHORT: got %.6f", sl)
	}
}

func TestEnforceMinSLDistanceLong(t *testing.T) {
	sl := EnforceMinSLDistance(100.0, 99.9, "LONG", 0.005, 0.01)
	if sl > 100.0*(1-0.005)+1e-6 {
		t.Fatalf("SL too close for LONG: got %.6f", sl)
	}
}
