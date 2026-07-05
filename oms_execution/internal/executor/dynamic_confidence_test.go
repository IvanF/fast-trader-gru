package executor

import (
	"math"
	"testing"

	"github.com/fast-trader-gru/oms_execution/internal/grid"
	"github.com/fast-trader-gru/oms_execution/internal/models"
)

func TestSymbolStatsRecord(t *testing.T) {
	ss := &SymbolStats{}

	ss.Record(0.10)
	if ss.TotalTrades != 1 || ss.Wins != 1 || ss.Losses != 0 || ss.ConsecLosses != 0 {
		t.Fatalf("after win: %+v", ss)
	}

	ss.Record(-0.05)
	if ss.TotalTrades != 2 || ss.Wins != 1 || ss.Losses != 1 || ss.ConsecLosses != 1 {
		t.Fatalf("after loss: %+v", ss)
	}

	ss.Record(-0.03)
	if ss.ConsecLosses != 2 {
		t.Fatalf("consec_losses should be 2, got %d", ss.ConsecLosses)
	}

	ss.Record(0.10)
	if ss.ConsecLosses != 0 {
		t.Fatalf("consec_losses should reset to 0, got %d", ss.ConsecLosses)
	}
}

func TestSymbolStatsWinRate(t *testing.T) {
	ss := &SymbolStats{}
	if wr := ss.WinRate(); wr != 0.5 {
		t.Fatalf("empty stats should return 0.5, got %f", wr)
	}

	ss.Record(1.0)
	ss.Record(1.0)
	ss.Record(-1.0)
	if wr := ss.WinRate(); math.Abs(wr-0.6667) > 0.01 {
		t.Fatalf("expected ~0.67, got %f", wr)
	}
}

func TestSymbolStatsPenalty(t *testing.T) {
	ss := &SymbolStats{}
	// < 3 trades: penalty = 1.0
	if p := ss.Penalty(); p != 1.0 {
		t.Fatalf("expected 1.0, got %f", p)
	}

	// 3 trades, WR=40%: penalty = max(1.0, 3.0 - 2.5*(0.40/0.40)) = 1.0
	ss.Record(1.0)
	ss.Record(1.0)
	ss.Record(-1.0)
	if p := ss.Penalty(); p != 1.0 {
		t.Fatalf("expected 1.0 for WR=67%%, got %f", p)
	}

	// 3 trades, WR=0%: penalty = max(1.0, 3.0 - 2.5*(0/0.40)) = 3.0
	// + 3 consec losses streak bonus: 3.0 * 1.6 = 4.8, capped at 3.75
	ss2 := &SymbolStats{}
	ss2.Record(-1.0)
	ss2.Record(-1.0)
	ss2.Record(-1.0)
	if p := ss2.Penalty(); p != 3.75 {
		t.Fatalf("expected 3.75 (3.0 + streak bonus, capped), got %f", p)
	}
}

func TestSymbolStatsConsecLossStreak(t *testing.T) {
	ss := &SymbolStats{}
	ss.Record(-1.0)
	ss.Record(-1.0) // 2 consec
	p := ss.Penalty()
	if p < 1.0 || p > 4.0 {
		t.Fatalf("streak=2 penalty should be > 1.0, got %f", p)
	}

	ss.Record(-1.0) // 3 consec
	p3 := ss.Penalty()
	if p3 <= p {
		t.Fatalf("streak=3 penalty (%f) should be > streak=2 (%f)", p3, p)
	}
}

func TestSymbolStatsEffectiveConfidence(t *testing.T) {
	ss := &SymbolStats{}
	ss.Record(-1.0)
	ss.Record(-1.0)
	ss.Record(-1.0) // 3 consec losses, WR=0%

	eff := ss.EffectiveConfidence(0.40)
	if eff > 0.95 {
		t.Fatalf("effective should cap at 0.95, got %f", eff)
	}
	if eff <= 0.40 {
		t.Fatalf("effective should be > base (0.40), got %f", eff)
	}
}

func TestTPDriftThreshold(t *testing.T) {
	// tpStillValid: threshold changed from 0.3% to 2%
	// price at 1.0, fresh TP at 1.01 → drift = 1% → should be valid (< 2%)
	existing := models.ExitOrder{Price: 1.0, Kind: "wall"}
	fresh := []grid.ExitLevel{{Price: 1.01, Kind: "wall"}}
	if !tpStillValid(existing, fresh) {
		t.Fatal("1%% drift should be valid with 2%% threshold")
	}

	// price at 1.0, fresh TP at 1.025 → drift = 2.5% → should be invalid (> 2%)
	fresh2 := []grid.ExitLevel{{Price: 1.025, Kind: "wall"}}
	if tpStillValid(existing, fresh2) {
		t.Fatal("2.5%% drift should be invalid with 2%% threshold")
	}
}
