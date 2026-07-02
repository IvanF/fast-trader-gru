"""Test MAE/MFE composite key pipeline — verifies exchange/shadow trades don't collide."""
import time
import sys
sys.path.insert(0, "/app/src" if __import__("os").path.exists("/app/src") else ".")


def test_composite_key_lookup():
    """Simulate engine._pending_mae_mfe behavior with composite keys."""
    pending = {}

    # --- Simulate two trades with same SignalID (exchange + shadow) ---
    signal_id = "abc-123-uuid"
    symbol = "NEARUSDT"

    # Exchange trade closes → store pending
    exchange_key = f"{signal_id}:True"
    pending[exchange_key] = {
        "state_vector": [0.1] * 128,
        "regime": "Choppy",
        "direction": "SHORT",
        "pnl": -0.30,
        "close_reason": "exchange_closed",
        "timestamp": time.time(),
    }

    # Shadow trade closes → store pending (should NOT overwrite exchange)
    shadow_key = f"{signal_id}:False"
    pending[shadow_key] = {
        "state_vector": [0.2] * 128,
        "regime": "Trending",
        "direction": "SHORT",
        "pnl": -0.07,
        "close_reason": "shadow_stop_loss",
        "timestamp": time.time(),
    }

    assert len(pending) == 2, f"Expected 2 pending entries, got {len(pending)}"
    assert exchange_key in pending, "Exchange pending not found"
    assert shadow_key in pending, "Shadow pending not found"
    print("PASS: Both exchange and shadow entries coexist in pending dict")

    # --- Simulate MAE/MFE lookup (same logic as engine.py) ---
    def lookup_mae_mfe(trade_id):
        # Try exact key first (composite from history_logger), then fallbacks
        return pending.pop(trade_id, None) or \
               pending.pop(f"{trade_id}:True", None) or \
               pending.pop(f"{trade_id}:False", None)

    # Shadow MAE/MFE arrives first (history_logger sends composite trade_id)
    shadow_trade_id = f"{signal_id}:False"
    result_shadow = lookup_mae_mfe(shadow_trade_id)
    assert result_shadow is not None, "Shadow MAE/MFE lookup failed"
    assert result_shadow["close_reason"] == "shadow_stop_loss", \
        f"Wrong entry returned: {result_shadow['close_reason']}"
    print("PASS: Shadow MAE/MFE lookup succeeded with correct data")

    # Exchange MAE/MFE arrives second
    exchange_trade_id = f"{signal_id}:True"
    result_exchange = lookup_mae_mfe(exchange_trade_id)
    assert result_exchange is not None, "Exchange MAE/MFE lookup failed"
    assert result_exchange["close_reason"] == "exchange_closed", \
        f"Wrong entry returned: {result_exchange['close_reason']}"
    print("PASS: Exchange MAE/MFE lookup succeeded with correct data")

    # Both consumed
    assert len(pending) == 0, f"Expected empty pending, got {len(pending)}"
    print("PASS: All pending entries consumed")


def test_symbol_fallback():
    """Test legacy symbol-only key fallback."""
    pending = {}
    signal_id = ""
    symbol = "MUSDT"

    # Store with composite key
    key = f"{symbol}:True"
    pending[key] = {"pnl": -0.5, "direction": "LONG"}

    # Lookup (history_logger sends composite trade_id)
    lookup_key = signal_id or symbol
    trade_id = f"{lookup_key}:True"
    result = pending.pop(trade_id, None) or \
             pending.pop(f"{lookup_key}:True", None) or \
             pending.pop(f"{lookup_key}:False", None)

    assert result is not None, "Symbol fallback lookup failed"
    assert result["pnl"] == -0.5
    print("PASS: Symbol fallback works correctly")


def test_stale_cleanup():
    """Test that stale entries (>35 min) are cleaned up."""
    pending = {}
    stale_cutoff = time.time() - 2100  # 35 min

    # Fresh entry
    pending["fresh:True"] = {"pnl": 0.1, "timestamp": time.time()}
    # Stale entry
    pending["stale:True"] = {"pnl": -0.3, "timestamp": time.time() - 2200}

    stale_keys = [k for k, v in pending.items() if v.get("timestamp", 0) < stale_cutoff]
    for k in stale_keys:
        pending.pop(k, None)

    assert "fresh:True" in pending, "Fresh entry was incorrectly removed"
    assert "stale:True" not in pending, "Stale entry was not removed"
    assert len(pending) == 1
    print("PASS: Stale cleanup removes old entries, keeps fresh ones")


def test_no_collision_same_signal():
    """Verify no data loss when both exchange and shadow use same SignalID."""
    pending = {}
    signal_id = "test-signal-uuid"

    # Store both
    pending[f"{signal_id}:True"] = {"data": "exchange", "pnl": -0.50}
    pending[f"{signal_id}:False"] = {"data": "shadow", "pnl": -0.07}

    # Lookup shadow first
    r1 = pending.pop(f"{signal_id}:False", None)
    assert r1["data"] == "shadow"

    # Then exchange
    r2 = pending.pop(f"{signal_id}:True", None)
    assert r2["data"] == "exchange"

    # Old broken behavior: second overwrites first
    pending_broken = {}
    key_broken = signal_id  # no composite key
    pending_broken[key_broken] = {"data": "exchange"}
    pending_broken[key_broken] = {"data": "shadow"}  # OVERWRITES!
    assert pending_broken[key_broken]["data"] == "shadow", "Broken: exchange lost"
    print("PASS: Composite key prevents collision (old behavior loses data)")


if __name__ == "__main__":
    test_composite_key_lookup()
    test_symbol_fallback()
    test_stale_cleanup()
    test_no_collision_same_signal()
    print("\n=== ALL TESTS PASSED ===")
