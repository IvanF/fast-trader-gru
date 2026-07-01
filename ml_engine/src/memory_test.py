"""Tests for ExperienceEngine memory query — matches bug fix."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from src.memory import ExperienceEngine


def _make_engine(entries: list[tuple[np.ndarray, float, str, str]] | None = None) -> ExperienceEngine:
    """Create a fresh in-memory ExperienceEngine, optionally pre-loaded."""
    import tempfile, os
    tmp = tempfile.mkdtemp()
    path = os.path.join(tmp, "test_index")
    eng = ExperienceEngine(dim=8, index_path=path, decay_days=14)
    if entries:
        for vec, pnl, regime, direction in entries:
            eng.add(vec, pnl, regime, direction)
    return eng


def test_empty_index_returns_zero_matches():
    eng = _make_engine()
    q = np.random.randn(8).astype(np.float32)
    _, info = eng.query(q, "Choppy")
    assert info["matches"] == 0
    assert info["long_matches"] == 0
    assert info["short_matches"] == 0


def test_matches_equals_long_plus_short():
    eng = _make_engine()
    v1 = np.ones(8, dtype=np.float32); v1[0] = 2.0
    v2 = np.ones(8, dtype=np.float32); v2[1] = 3.0
    eng.add(v1, 0.01, "Choppy", "LONG")
    eng.add(v2, -0.02, "Choppy", "SHORT")
    eng.add(v1, 0.005, "Choppy", "LONG")

    q = np.ones(8, dtype=np.float32); q[0] = 1.5
    _, info = eng.query(q, "Choppy", k=10)
    assert info["matches"] == info["long_matches"] + info["short_matches"]


def test_matches_not_always_k():
    """Before the fix, matches was always k regardless of content."""
    eng = _make_engine()
    v = np.ones(8, dtype=np.float32); v[0] = 5.0
    eng.add(v, 0.01, "Choppy", "LONG")
    eng.add(v, 0.02, "Choppy", "LONG")

    q = np.ones(8, dtype=np.float32); q[0] = 4.0
    _, info = eng.query(q, "Choppy", k=10)
    # All entries are LONG, so short_matches=0
    assert info["short_matches"] == 0
    assert info["matches"] == info["long_matches"]
    # matches should be ≤ 2 (only 2 entries exist), NOT 10
    assert info["matches"] <= 2


def test_all_short():
    eng = _make_engine()
    v = np.ones(8, dtype=np.float32); v[2] = 1.0
    eng.add(v, -0.01, "Choppy", "SHORT")
    eng.add(v, -0.005, "Choppy", "SHORT")

    q = np.ones(8, dtype=np.float32); q[2] = 0.8
    _, info = eng.query(q, "Choppy", k=10)
    assert info["long_matches"] == 0
    assert info["short_matches"] >= 1
    assert info["matches"] == info["short_matches"]


def test_long_short_split():
    eng = _make_engine()
    v = np.ones(8, dtype=np.float32)
    eng.add(v, 0.01, "Choppy", "LONG")
    eng.add(v, -0.01, "Choppy", "SHORT")

    _, info = eng.query(v, "Choppy", k=10)
    assert info["long_matches"] + info["short_matches"] == info["matches"]


def _run_all():
    tests = [
        test_empty_index_returns_zero_matches,
        test_matches_equals_long_plus_short,
        test_matches_not_always_k,
        test_all_short,
        test_long_short_split,
    ]
    for t in tests:
        t()
        print(f"  PASS: {t.__name__}")
    print(f"\n{len(tests)}/{len(tests)} tests passed")


if __name__ == "__main__":
    _run_all()
