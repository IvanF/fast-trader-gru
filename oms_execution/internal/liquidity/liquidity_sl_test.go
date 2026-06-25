package liquidity

import (
	"testing"

	"github.com/fast-trader-gru/oms_execution/internal/models"
)

func TestComputeLiquiditySL_LONG_Zone(t *testing.T) {
	ob := models.OrderbookSnapshot{
		Bids: []models.OrderbookLevel{
			{Price: "99.90", Size: "2"},
			{Price: "99.85", Size: "1"},
			{Price: "99.80", Size: "3"},
			{Price: "99.50", Size: "50"},
			{Price: "99.00", Size: "5"},
		},
	}
	result := ComputeLiquiditySL("LONG", 100.0, ob, 0.01, 0.003, 0.01)
	t.Logf("SL=%.2f dist=%.3f%% level=%.2f(size=%.1f) source=%s",
		result.Price, result.DistancePct*100, result.LevelPrice, result.LevelSize, result.Source)
	if result.Price >= 100.0 {
		t.Errorf("SL must be below entry")
	}
	if result.DistancePct < 0.003 {
		t.Errorf("SL too close: %.4f%%", result.DistancePct*100)
	}
}

func TestComputeLiquiditySL_SHORT_Zone(t *testing.T) {
	ob := models.OrderbookSnapshot{
		Asks: []models.OrderbookLevel{
			{Price: "100.10", Size: "1"},
			{Price: "100.15", Size: "2"},
			{Price: "100.20", Size: "3"},
			{Price: "100.50", Size: "40"},
			{Price: "101.00", Size: "5"},
		},
	}
	result := ComputeLiquiditySL("SHORT", 100.0, ob, 0.01, 0.003, 0.01)
	t.Logf("SL=%.2f dist=%.3f%% level=%.2f(size=%.1f) source=%s",
		result.Price, result.DistancePct*100, result.LevelPrice, result.LevelSize, result.Source)
	if result.Price <= 100.0 {
		t.Errorf("SL must be above entry")
	}
	if result.DistancePct < 0.003 {
		t.Errorf("SL too close: %.4f%%", result.DistancePct*100)
	}
}

func TestComputeLiquiditySL_EmptyBook(t *testing.T) {
	result := ComputeLiquiditySL("LONG", 100.0, models.OrderbookSnapshot{}, 0.01, 0.003, 0.01)
	t.Logf("SL=%.2f source=%s", result.Price, result.Source)
	if result.Price >= 100.0 {
		t.Errorf("SL must be below entry")
	}
}

func TestComputeLiquiditySL_ThinLevels_FoundZone(t *testing.T) {
	// Multiple thin levels close together → should form a zone
	ob := models.OrderbookSnapshot{
		Bids: []models.OrderbookLevel{
			{Price: "99.95", Size: "0.5"},
			{Price: "99.90", Size: "0.5"},
			{Price: "99.85", Size: "0.5"},
			{Price: "99.80", Size: "0.5"},
			{Price: "99.75", Size: "0.5"},
		},
	}
	result := ComputeLiquiditySL("LONG", 100.0, ob, 0.01, 0.003, 0.01)
	t.Logf("SL=%.2f dist=%.3f%% source=%s", result.Price, result.DistancePct*100, result.Source)
	if result.Price >= 100.0 {
		t.Errorf("SL must be below entry")
	}
}

func TestComputeLiquiditySL_FarLevel_Fallback(t *testing.T) {
	ob := models.OrderbookSnapshot{
		Bids: []models.OrderbookLevel{
			{Price: "95.0", Size: "1"},
		},
	}
	result := ComputeLiquiditySL("LONG", 100.0, ob, 0.01, 0.003, 0.01)
	t.Logf("SL=%.2f dist=%.3f%% source=%s", result.Price, result.DistancePct*100, result.Source)
	if result.DistancePct < 0.003 {
		t.Errorf("SL too close: %.4f%%", result.DistancePct*100)
	}
}

func TestComputeLiquiditySL_DenseCluster(t *testing.T) {
	ob := models.OrderbookSnapshot{
		Asks: []models.OrderbookLevel{
			{Price: "100.10", Size: "100"},
			{Price: "100.20", Size: "200"},
			{Price: "100.30", Size: "150"},
			{Price: "102.00", Size: "10"},
		},
	}
	result := ComputeLiquiditySL("SHORT", 100.0, ob, 0.01, 0.003, 0.01)
	t.Logf("SL=%.2f dist=%.3f%% level=%.2f(size=%.1f) source=%s",
		result.Price, result.DistancePct*100, result.LevelPrice, result.LevelSize, result.Source)
	if result.Price <= 100.0 {
		t.Errorf("SL must be above entry")
	}
}
