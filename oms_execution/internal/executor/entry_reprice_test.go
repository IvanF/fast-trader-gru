package executor

import "testing"

func TestEntryOrderIsDead(t *testing.T) {
	if !entryOrderIsDead("Cancelled") {
		t.Fatal("Cancelled should be dead")
	}
	if entryOrderIsDead("Filled") {
		t.Fatal("Filled should not be dead")
	}
}

func TestEntryOrderIsExecuted(t *testing.T) {
	if !entryOrderIsExecuted("Filled", 0) {
		t.Fatal("Filled should be executed")
	}
	if !entryOrderIsExecuted("PartiallyFilled", 1) {
		t.Fatal("PartiallyFilled with qty should be executed")
	}
	if entryOrderIsExecuted("PartiallyFilled", 0) {
		t.Fatal("empty partial should not be executed")
	}
}

func TestPositionSizeViolation(t *testing.T) {
	if !positionSizeViolation(188, 94) {
		t.Fatal("188 > 94*1.5 should violate")
	}
	if positionSizeViolation(140, 94) {
		t.Fatal("140 <= 141 should not violate")
	}
	if positionSizeViolation(100, 0) {
		t.Fatal("zero target should not violate")
	}
}
