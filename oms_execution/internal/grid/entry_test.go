package grid

import (
	"testing"

	"github.com/fast-trader-gru/oms_execution/internal/models"
)

func TestAggressiveMakerEntry(t *testing.T) {
	ob := models.OrderbookSnapshot{
		Bids: []models.OrderbookLevel{{Price: "100.0", Size: "10"}},
		Asks: []models.OrderbookLevel{{Price: "100.10", Size: "10"}},
	}
	longEntry := AggressiveMakerEntry("LONG", ob, 0.01, 2)
	if longEntry < 100.0 || longEntry >= 100.10 {
		t.Fatalf("LONG entry=%f want in [100.0, 100.10)", longEntry)
	}

	shortEntry := AggressiveMakerEntry("SHORT", ob, 0.01, 2)
	if shortEntry <= 100.0 || shortEntry > 100.10 {
		t.Fatalf("SHORT entry=%f want in (100.0, 100.10]", shortEntry)
	}

	if AggressiveMakerEntry("LONG", ob, 0.01, 0) != 0 {
		t.Fatal("makerTicks=0 should return 0")
	}
}

func TestPassiveMakerExitPrice(t *testing.T) {
	ob := models.OrderbookSnapshot{
		Bids: []models.OrderbookLevel{{Price: "100.0", Size: "10"}},
		Asks: []models.OrderbookLevel{{Price: "100.10", Size: "10"}},
	}
	longExit := PassiveMakerExitPrice("LONG", ob, 0.01, 2)
	if longExit <= 100.0 || longExit > 100.10 {
		t.Fatalf("LONG exit=%f want in (100.0, 100.10]", longExit)
	}
	shortExit := PassiveMakerExitPrice("SHORT", ob, 0.01, 2)
	if shortExit < 100.0 || shortExit >= 100.10 {
		t.Fatalf("SHORT exit=%f want in [100.0, 100.10)", shortExit)
	}
}

func TestCapVolMultiplier(t *testing.T) {
	if CapVolMultiplier(3.0, 1.5) != 1.5 {
		t.Fatal("expected cap at 1.5")
	}
	if CapVolMultiplier(0, 1.5) != 1.0 {
		t.Fatal("expected default 1.0 for zero vm")
	}
}
