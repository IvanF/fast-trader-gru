"""Tests for Config loading — confidence threshold values."""

import os
import sys

# Ensure we're importing from the right place
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.config import Config


def test_confidence_threshold_from_env():
    os.environ["CONFIDENCE_THRESHOLD"] = "0.15"
    cfg = Config.load()
    assert cfg.confidence_threshold == 0.15, f"expected 0.15, got {cfg.confidence_threshold}"


def test_long_confidence_threshold_from_env():
    os.environ["LONG_CONFIDENCE_THRESHOLD"] = "0.15"
    cfg = Config.load()
    assert cfg.long_confidence_threshold == 0.15, f"expected 0.15, got {cfg.long_confidence_threshold}"


def test_default_fallback():
    os.environ.pop("CONFIDENCE_THRESHOLD", None)
    os.environ.pop("LONG_CONFIDENCE_THRESHOLD", None)
    cfg = Config.load()
    assert cfg.confidence_threshold == 0.30, f"expected fallback 0.30, got {cfg.confidence_threshold}"
    assert cfg.long_confidence_threshold == 0.50, f"expected fallback 0.50, got {cfg.long_confidence_threshold}"


def _run_all():
    tests = [
        test_confidence_threshold_from_env,
        test_long_confidence_threshold_from_env,
        test_default_fallback,
    ]
    for t in tests:
        t()
        print(f"  PASS: {t.__name__}")
    print(f"\n{len(tests)}/{len(tests)} tests passed")


if __name__ == "__main__":
    _run_all()
