package executor

import "testing"

func TestSlWouldWidenLong(t *testing.T) {
	if !slWouldWiden("LONG", 100, 99) {
		t.Fatal("lower SL should widen LONG stop")
	}
	if slWouldWiden("LONG", 100, 101) {
		t.Fatal("higher SL should tighten LONG stop")
	}
}

func TestSlWouldWidenShort(t *testing.T) {
	if !slWouldWiden("SHORT", 100, 101) {
		t.Fatal("higher SL should widen SHORT stop")
	}
	if slWouldWiden("SHORT", 100, 99) {
		t.Fatal("lower SL should tighten SHORT stop")
	}
}

func TestClampSLTightenOnly(t *testing.T) {
	if got := clampSLTightenOnly("LONG", 100, 99); got != 100 {
		t.Fatalf("LONG clamp = %v, want 100", got)
	}
	if got := clampSLTightenOnly("LONG", 100, 101); got != 101 {
		t.Fatalf("LONG tighten = %v, want 101", got)
	}
	if got := clampSLTightenOnly("SHORT", 100, 101); got != 100 {
		t.Fatalf("SHORT clamp = %v, want 100", got)
	}
	if got := clampSLTightenOnly("SHORT", 100, 99); got != 99 {
		t.Fatalf("SHORT tighten = %v, want 99", got)
	}
}
