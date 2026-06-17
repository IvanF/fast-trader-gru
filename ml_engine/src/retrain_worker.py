"""Rolling retraining worker — 2h interval or 100-trade trigger."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

from .config import Config
from . import metrics as prom

logger = logging.getLogger(__name__)

REPORT_PATH = "/app/data/retrain_report.json"


class RollingRetrainWorker:
    """Spawns low-priority train.py without blocking the async inference loop."""

    def __init__(self, cfg: Config, symbols_provider) -> None:
        self.cfg = cfg
        self._symbols_provider = symbols_provider
        self._lock = asyncio.Lock()
        self._running = False
        self._trades_since_retrain = 0
        self._last_retrain_at = time.time()

    @property
    def trades_since_retrain(self) -> int:
        return self._trades_since_retrain

    def log_schedule(self) -> None:
        logger.info(
            "retrain scheduled: every %.0fh or after %d closed trades (lookback=%dh epochs=%d)",
            self.cfg.retrain_interval_sec / 3600,
            self.cfg.retrain_trade_threshold,
            self.cfg.retrain_lookback_hours,
            self.cfg.retrain_epochs,
        )

    async def on_trade_closed(self) -> None:
        self._trades_since_retrain += 1
        prom.retrain_trades_since_last.set(self._trades_since_retrain)
        if self._trades_since_retrain >= self.cfg.retrain_trade_threshold:
            logger.info(
                "retrain trigger: %d trades reached threshold %d",
                self._trades_since_retrain,
                self.cfg.retrain_trade_threshold,
            )
            await self.trigger("trade_threshold")

    async def interval_loop(self) -> None:
        while True:
            await asyncio.sleep(self.cfg.retrain_interval_sec)
            elapsed = time.time() - self._last_retrain_at
            if elapsed >= self.cfg.retrain_interval_sec:
                logger.info("retrain trigger: %.0fs interval elapsed", elapsed)
                await self.trigger("interval")

    async def trigger(self, reason: str) -> None:
        if self._running:
            logger.info("retrain skipped (%s): already running", reason)
            return
        async with self._lock:
            if self._running:
                return
            self._running = True
        try:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, self._run_train_subprocess, reason)
        finally:
            self._running = False
            self._trades_since_retrain = 0
            self._last_retrain_at = time.time()
            prom.retrain_trades_since_last.set(0)

    def _run_train_subprocess(self, reason: str) -> None:
        symbols = self._symbols_provider()
        if not symbols:
            symbols = os.getenv("RETRAIN_DEFAULT_SYMBOLS", "BTCUSDT,ETHUSDT")
        symbol_arg = ",".join(symbols) if isinstance(symbols, (list, set)) else str(symbols)

        train_script = Path(__file__).resolve().parents[1] / "scripts" / "train.py"
        cmd = [
            "nice", "-n", str(self.cfg.train_nice_level),
            sys.executable,
            str(train_script),
            "--symbols", symbol_arg,
            "--hours", str(self.cfg.retrain_lookback_hours),
            "--epochs", str(self.cfg.retrain_epochs),
            "--trigger", reason,
            "--incremental",
            "--device", self.cfg.train_device,
        ]

        start = time.time()
        prom.retrain_running.set(1)
        logger.info("starting retrain subprocess: %s", " ".join(cmd))

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.cfg.retrain_timeout_sec,
                env={**os.environ, "MODEL_DIR": self.cfg.model_dir, "TRAIN_DEVICE": self.cfg.train_device},
            )
            duration = time.time() - start
            prom.retrain_duration.observe(duration)
            self._apply_report(result.returncode, duration)

            if result.returncode == 0:
                logger.info("retrain completed in %.1fs reason=%s", duration, reason)
                if result.stdout:
                    logger.info("train stdout:\n%s", result.stdout[-2000:])
            elif result.returncode == 3:
                logger.warning(
                    "retrain skipped (insufficient InfluxDB samples — need closed trades): reason=%s\n%s",
                    reason,
                    (result.stdout or result.stderr)[-1000:],
                )
            elif result.returncode == 4:
                prom.retrain_failures.inc()
                logger.error(
                    "retrain failed: CUDA not available (TRAIN_DEVICE=cuda). "
                    "Ensure gpus: all and nvidia-container-toolkit.\n%s",
                    (result.stderr or result.stdout)[-1000:],
                )
            else:
                prom.retrain_failures.inc()
                logger.error(
                    "retrain failed code=%d reason=%s\nstdout=%s\nstderr=%s",
                    result.returncode, reason, result.stdout[-1000:], result.stderr[-1000:],
                )
        except subprocess.TimeoutExpired:
            prom.retrain_failures.inc()
            prom.retrain_duration.observe(time.time() - start)
            logger.error("retrain timed out after %ds", self.cfg.retrain_timeout_sec)
        except FileNotFoundError:
            # nice not available — fall back to direct python with os.nice inside train.py
            cmd_fallback = cmd[2:]  # drop nice -n 15
            result = subprocess.run(
                cmd_fallback,
                capture_output=True,
                text=True,
                timeout=self.cfg.retrain_timeout_sec,
                env={**os.environ, "MODEL_DIR": self.cfg.model_dir, "TRAIN_DEVICE": self.cfg.train_device},
            )
            duration = time.time() - start
            prom.retrain_duration.observe(duration)
            self._apply_report(result.returncode, duration)
        finally:
            prom.retrain_running.set(0)

    def _apply_report(self, exit_code: int, duration: float) -> None:
        report: dict = {}
        if os.path.exists(REPORT_PATH):
            try:
                with open(REPORT_PATH, "r", encoding="utf-8") as f:
                    report = json.load(f)
            except (json.JSONDecodeError, OSError):
                pass

        if exit_code == 0 and report:
            prom.loss_delta.set(float(report.get("loss_delta", 0)))
            prom.last_training_timestamp.set(float(report.get("timestamp", time.time())))
            version = str(report.get("version", "unknown"))
            prom.set_model_version_info(version, float(report.get("timestamp", time.time())))
            logger.info("retrain metrics: loss_delta=%.4f version=%s", report.get("loss_delta", 0), version)

        report.setdefault("duration_sec", duration)
        report.setdefault("exit_code", exit_code)
        report.setdefault("timestamp", time.time())
