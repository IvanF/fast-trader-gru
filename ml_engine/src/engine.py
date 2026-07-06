"""Async ML engine core — Redis listener, inference loop, continuous learning."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import threading
import time
import uuid
from typing import Dict, List, Set, TypedDict

import msgpack
import numpy as np
import redis.asyncio as aioredis

from .config import Config
from .features import FeatureStore
from .inference import HotSwapONNXInference
from .influx_store import InfluxStore
from .memory import ExperienceEngine
from .model_manifest import ModelManifest
from . import metrics as prom
from .market_telemetry import EventStateStore, compute_telemetry_and_events
from .movement_probability import MovementStateStore, load_weights_from_env, softmax_movement_probability
from .regime import RegimeDetector
from .exit_plan import build_exit_plan
from .online_learner import OnlineLearner, CONSECUTIVE_LOSS_THRESHOLD
from .retrain_worker import RollingRetrainWorker
from .train_device import log_cuda_status

logger = logging.getLogger(__name__)


class _OpenPosition(TypedDict):
    direction: str
    signal_id: str
    entry_confidence: float
    decay_signaled: bool
    adverse_streak: int


class _PendingEntry(TypedDict):
    direction: str
    signal_id: str
    entry_confidence: float
    abort_signaled: bool
    adverse_streak: int


class MLEngine:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.features = FeatureStore(window_sec=cfg.buffer_seconds)
        self.regime = RegimeDetector()
        self.memory = ExperienceEngine(cfg.state_dim, cfg.faiss_path, cfg.memory_decay_days)
        self.inference = HotSwapONNXInference(cfg.model_dir, cfg.state_dim)
        self.influx: InfluxStore | None = None
        if cfg.influx_token:
            self.influx = InfluxStore(
                cfg.influx_url, cfg.influx_token, cfg.influx_org,
                cfg.influx_bucket_raw, cfg.influx_bucket_features,
            )
        self.active_symbols: Set[str] = set()
        self._redis: aioredis.Redis | None = None
        self._last_regime_run: Dict[str, float] = {}
        self._funding_rates: Dict[str, float] = {}
        self._stats: Dict[str, int] = {
            "market_events": 0,
            "orderbook_events": 0,
            "trade_events": 0,
            "predictions": 0,
            "signals": 0,
            "hold_low_conf": 0,
            "hold_direction": 0,
            "hold_correlation": 0,
            "buffer_warming": 0,
            "pattern_blocks": 0,
        }
        self._last_status_log = time.time()
        self._pattern_block_count_this_cycle = 0
        self._tick_cycle_predictions = 0
        self._circuit_breaker_lock = threading.Lock()
        self._symbol_cooldowns: dict[str, float] = {}  # symbol -> expire timestamp
        self._toxic_prob_cache: float = 0.0  # toxic flow prob from last inference
        self._symbol_setup_losses: dict[str, list[float]] = {}  # "SYMBOL_DIR_REGIME" -> [pnl1, pnl2, ...]
        self._corr_block_count_this_cycle = 0
        self._last_signal_at = time.time()
        self._signal_drought_warned = False
        self._online_learner = OnlineLearner(cfg.model_dir, cfg.state_dim)
        self._consecutive_losses = 0
        self._symbol_trades: Dict[str, Dict[str, int]] = {}  # symbol -> {"wins": N, "losses": N}
        self._retrain_worker = RollingRetrainWorker(
            cfg, symbols_provider=lambda: sorted(self.active_symbols),
        )
        self._open_positions: Dict[str, _OpenPosition] = {}
        self._pending_entries: Dict[str, _PendingEntry] = {}
        self._pending_mae_mfe: Dict[str, dict] = {}
        self._event_states = EventStateStore()
        self._movement_states = MovementStateStore()
        self._movement_weights, self._movement_bias = load_weights_from_env()
        self._movement_funding_prior = float(os.getenv("MOVEMENT_PROB_FUNDING_PRIOR_SCALE", "50"))
        self._started_at = time.time()
        self._last_telemetry: Dict[str, float] = {}

    async def connect_redis(self) -> None:
        backoff = 1.0
        while True:
            try:
                self._redis = aioredis.Redis(
                    host=self.cfg.redis_host,
                    port=self.cfg.redis_port,
                    decode_responses=False,
                    socket_connect_timeout=5,
                )
                await self._redis.ping()
                logger.info("connected to redis at %s", self.cfg.redis_addr)
                await self._bootstrap_active_symbols()
                return
            except Exception as exc:
                logger.warning("redis connect failed: %s, retry in %.1fs", exc, backoff)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 30.0)

    async def _new_redis(self) -> aioredis.Redis:
        conn = aioredis.Redis(
            host=self.cfg.redis_host,
            port=self.cfg.redis_port,
            decode_responses=False,
            socket_connect_timeout=5,
            socket_timeout=5,
        )
        await conn.ping()
        return conn

    async def _shutdown_redis(
        self,
        conn: aioredis.Redis | None,
        pubsub: aioredis.client.PubSub | None,
        channels: list[str] | None = None,
    ) -> None:
        if pubsub is not None:
            if channels:
                try:
                    await pubsub.unsubscribe(*channels)
                except Exception:
                    pass
            try:
                await asyncio.wait_for(pubsub.aclose(), timeout=1.0)
            except Exception:
                pass
        if conn is not None:
            try:
                await asyncio.wait_for(conn.aclose(), timeout=1.0)
            except Exception:
                pass

    async def _pubsub_listener(self, name: str, channel: str, handler) -> None:
        """Dedicated Redis connection per pub/sub listener."""
        while True:
            conn: aioredis.Redis | None = None
            pubsub = None
            try:
                conn = await self._new_redis()
                pubsub = conn.pubsub()
                await pubsub.subscribe(channel)
                logger.info("%s listener subscribed to %s", name, channel)
                while True:
                    msg = await pubsub.get_message(ignore_subscribe_messages=True, timeout=1.0)
                    if msg is None:
                        continue
                    if msg["type"] != "message":
                        logger.debug("%s got non-message type: %s", name, msg["type"])
                        continue
                    logger.info("%s handler called, channel=%s data_len=%d", name, msg.get("channel", "?"), len(str(msg.get("data", ""))))
                    await handler(msg)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.error("%s listener error: %s", name, exc)
                await asyncio.sleep(2)
            finally:
                await self._shutdown_redis(conn, pubsub, [channel])

    async def _bootstrap_active_symbols(self) -> None:
        assert self._redis is not None
        raw = await self._redis.get("config:active_symbols:latest")
        if not raw:
            logger.info("no cached active symbols yet — waiting for screener")
            return
        if isinstance(raw, bytes):
            raw = raw.decode()
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            logger.warning("invalid cached active symbols: %s", exc)
            return
        self._apply_active_symbols(data)

    def _apply_active_symbols(self, data: dict) -> None:
        symbols = [
            sym for sym in data.get("symbols", [])
            if sym not in self.cfg.blacklist_symbols
        ]
        self.active_symbols = set(symbols)
        for meta in data.get("meta", []):
            sym = meta.get("symbol")
            if sym:
                self._funding_rates[sym] = float(meta.get("funding_rate", 0))
                self.features.set_funding_rate(sym, self._funding_rates[sym])
        sample = ",".join(sorted(self.active_symbols)[:5])
        suffix = "..." if len(self.active_symbols) > 5 else ""
        logger.info(
            "active symbols updated: %d [%s%s]",
            len(self.active_symbols),
            sample,
            suffix,
        )

    async def run(self) -> None:
        await self.connect_redis()
        assert self._redis is not None

        # Restore per-symbol WR from Redis
        try:
            keys = []
            async for key in self._redis.scan_iter("symbol_trades:*"):
                sym = key.decode().replace("symbol_trades:", "")
                data = json.loads(await self._redis.get(key))
                self._symbol_trades[sym] = data
            if self._symbol_trades:
                logger.info("restored symbol_trades from redis: %d symbols", len(self._symbol_trades))
            async for key in self._redis.scan_iter("symbol_setup_losses:*"):
                skey = key.decode().replace("symbol_setup_losses:", "")
                data = json.loads(await self._redis.get(key))
                self._symbol_setup_losses[skey] = data
            if self._symbol_setup_losses:
                logger.info("restored symbol_setup_losses from redis: %d setups", len(self._symbol_setup_losses))
        except Exception:
            pass

        port = int(self.cfg.metrics_addr.split(":")[-1])
        prom.start_metrics_server(port)
        health_port = port + 1000
        prom.register_health_check(self._health_check)
        prom.register_exit_plan_handler(self._handle_exit_plan_request)
        prom.start_health_server(health_port)
        prom.faiss_index_size.set(self.memory.size)
        manifest = ModelManifest.from_dir(self.cfg.model_dir)
        prom.set_model_version(manifest.version, manifest.updated_at)
        prom.set_model_version_info(manifest.version, manifest.updated_at)

        if self.influx is not None:
            self.influx.connect()

        log_cuda_status(logger)

        logger.info(
            "ml_engine started model=%s confidence_threshold=%.2f kill_confidence=%.2f entry_abort=%.2f vol_cap=%.2f entry_maker_ticks=%d "
            "retrain_every=%dh/%d_trades faiss_size=%d open_positions=%d pending_entries=%d",
            manifest.version,
            self.cfg.confidence_threshold,
            self.cfg.kill_confidence_threshold,
            self.cfg.entry_abort_threshold,
            self.cfg.vol_multiplier_cap,
            self.cfg.entry_maker_ticks,
            self.cfg.retrain_interval_sec // 3600,
            self.cfg.retrain_trade_threshold,
            self.memory.size,
            len(self._open_positions),
            len(self._pending_entries),
        )
        self._retrain_worker.log_schedule()

        await asyncio.gather(
            self._listen_active_symbols(),
            self._listen_market_data(),
            self._listen_execution_results(),
            self._listen_mae_mfe(),
            self._listen_positions(),
            self._listen_pending_orders(),
            self._listen_model_reload(),
            self._model_watch_loop(),
            self._retrain_worker.interval_loop(),
            self._regime_loop(),
            self._persist_loop(),
            self._metrics_loop(),
            self._status_loop(),
        )

    def _health_check(self) -> dict:
        checks = {}

        redis_ok = self._redis is not None
        checks["redis"] = "ok" if redis_ok else "disconnected"

        faiss_size = self.memory.size
        checks["faiss"] = "ok" if faiss_size > 0 else "empty"

        model_ok = self.inference._bundle.mlp is not None
        checks["model"] = "ok" if model_ok else "missing"

        checks["symbols"] = len(self.active_symbols)
        checks["open_positions"] = len(self._open_positions)
        checks["uptime_sec"] = int(time.time() - self._started_at) if hasattr(self, "_started_at") else 0

        healthy = redis_ok and model_ok
        return {"status": "ok" if healthy else "degraded", **checks}

    def _handle_exit_plan_request(self, request: dict) -> dict:
        direction = request.get("direction", "SHORT")
        bids = request.get("bids", [])
        asks = request.get("asks", [])
        tick_size = request.get("tick_size", 0.0001)
        vol_mult = request.get("volatility_multiplier", 1.0)
        regime = request.get("regime", "Choppy")
        confidence = request.get("confidence", self.cfg.confidence_threshold)

        if not bids or not asks:
            return {"error": "empty orderbook"}

        mid = (float(bids[0].get("price", 0)) + float(asks[0].get("price", 0))) / 2.0
        if mid <= 0:
            return {"error": "invalid mid price"}

        exit_sl = self.cfg.max_sl_pct
        exit_tp = self.cfg.max_tp_pct

        plan = build_exit_plan(
            direction,
            bids,
            asks,
            vol_mult,
            regime,
            confidence,
            tick_size=tick_size,
            fallback_mid=mid,
            vol_multiplier_cap=self.cfg.vol_multiplier_cap,
            entry_maker_ticks=self.cfg.entry_maker_ticks,
            min_tp_pct=exit_tp * 0.8,
            fee_breakeven_pct=self.cfg.fee_breakeven_pct,
            max_tp_pct=exit_tp,
            max_sl_pct=exit_sl,
        )
        return plan or {"error": "exit_plan_computation_failed"}

    async def _listen_active_symbols(self) -> None:
        async def handle(msg: dict) -> None:
            data = json.loads(msg["data"])
            self._apply_active_symbols(data)

        await self._pubsub_listener("active symbols", self.cfg.active_symbols_channel, handle)

    async def _listen_market_data(self) -> None:
        pubsub = None
        conn: aioredis.Redis | None = None
        subscribed: Set[str] = set()
        while True:
            try:
                if not self.active_symbols:
                    if time.time() - self._last_status_log > 30:
                        logger.info("waiting for active symbols from screener...")
                        self._last_status_log = time.time()
                    await asyncio.sleep(1)
                    continue

                desired = set()
                for sym in self.active_symbols:
                    desired.add(f"market:orderbook:{sym}")
                    desired.add(f"market:trades:{sym}")

                if desired != subscribed:
                    await self._shutdown_redis(conn, pubsub, sorted(subscribed) if subscribed else None)
                    pubsub = None
                    conn = None
                    conn = await self._new_redis()
                    pubsub = conn.pubsub()
                    await pubsub.subscribe(*sorted(desired))
                    subscribed = desired
                    logger.info("subscribed to %d market channels", len(desired))

                msg = await pubsub.get_message(ignore_subscribe_messages=True, timeout=1.0)
                if msg is None:
                    continue
                if msg["type"] != "message":
                    continue
                recv_at = time.time()
                channel = msg["channel"]
                if isinstance(channel, bytes):
                    channel = channel.decode()
                payload = self._decode_payload(msg["data"])
                await self._process_market_event(channel, payload, recv_at)
            except Exception as exc:
                logger.exception("market data listener error: %s", exc)
                subscribed = set()
                await self._shutdown_redis(conn, pubsub, None)
                pubsub = None
                conn = None
                await asyncio.sleep(2)

    async def _listen_execution_results(self) -> None:
        async def handle(msg: dict) -> None:
            raw = msg["data"]
            if isinstance(raw, str):
                data = json.loads(raw)
            elif isinstance(raw, dict):
                data = raw
            elif isinstance(raw, bytes):
                data = json.loads(raw.decode())
            else:
                return
            symbol = data.get("symbol", "")
            pnl = float(data.get("net_pnl", 0))
            direction = data.get("direction", "HOLD")
            logger.info("execution result received: %s %s pnl=%.4f", symbol, direction, pnl)

            # Track per-symbol WR BEFORE early returns
            if symbol and pnl != 0:
                self._symbol_trades.setdefault(symbol, {"wins": 0, "losses": 0})
                if pnl < 0:
                    self._symbol_trades[symbol]["losses"] += 1
                else:
                    self._symbol_trades[symbol]["wins"] += 1
                try:
                    result = await self._redis.setex(f"symbol_trades:{symbol}", 86400, json.dumps(self._symbol_trades[symbol]))
                    logger.info("symbol_trades saved: %s wins=%d losses=%d", symbol, self._symbol_trades[symbol].get("wins",0), self._symbol_trades[symbol].get("losses",0))
                except Exception as e:
                    logger.warning("symbol_trades persist failed for %s: %s (redis=%s)", symbol, e, self._redis)

            vec = np.array(data.get("state_vector", []), dtype=np.float32)
            if vec.ndim == 0 or vec.size == 0:
                return
            if vec.size < self.cfg.state_dim:
                vec = np.pad(vec, (0, self.cfg.state_dim - vec.size))
            pnl = float(data.get("net_pnl", 0))
            regime = data.get("regime", "Choppy")
            direction = data.get("direction", "HOLD")
            symbol = data.get("symbol", "")
            signal_id = data.get("signal_id", "")
            mae_pct = float(data.get("mae_pct", 0))
            mfe_pct = float(data.get("mfe_pct", 0))
            exchange_pnl = data.get("exchange_pnl", True)
            weight = 1.0 if exchange_pnl else 0.3

            # Don't add to FAISS here — wait for MAE/MFE from PriceTracker.
            # Store state_vector in Redis for persistence across restarts.
            # Normalize exchange_pnl to lowercase for consistent key matching
            ep_str = "true" if exchange_pnl else "false"
            lookup_key = f"{signal_id}:{ep_str}" if signal_id else f"{symbol}:{ep_str}"
            pending_data = {
                "state_vector": vec[: self.cfg.state_dim].tolist(),
                "regime": regime,
                "direction": direction,
                "symbol": symbol,
                "pnl": pnl,
                "close_reason": data.get("close_reason", ""),
                "exchange_pnl": exchange_pnl,
                "timestamp": time.time(),
            }
            self._pending_mae_mfe[lookup_key] = pending_data
            try:
                await self._redis.setex(
                    f"mae_mfe_pending:{lookup_key}",
                    2100,
                    json.dumps(pending_data),
                )
            except Exception:
                pass
            prom.faiss_index_size.set(self.memory.size)

            dir_idx = {"LONG": 0, "SHORT": 1}.get(direction.upper(), 2)
            self._online_learner.update(
                v_state=vec[: self.cfg.state_dim],
                v_memory=np.zeros(8, dtype=np.float32),
                direction=dir_idx,
                confidence=float(data.get("confidence", 0.5)),
                pnl=pnl,
                regime=regime,
                symbol=symbol,
            )

            if pnl < 0:
                self._consecutive_losses += 1
                cooldown = 1800 if self._consecutive_losses < 2 else 3600
                self._symbol_cooldowns[symbol] = time.time() + cooldown

                setup_key = f"{symbol}_{direction}_{regime}"
                self._symbol_setup_losses.setdefault(setup_key, []).append(pnl)
                if len(self._symbol_setup_losses[setup_key]) > 50:
                    self._symbol_setup_losses[setup_key] = self._symbol_setup_losses[setup_key][-50:]
                try:
                    await self._redis.setex(
                        f"symbol_setup_losses:{setup_key}", 86400,
                        json.dumps(self._symbol_setup_losses[setup_key]),
                    )
                except Exception:
                    pass

                if self._consecutive_losses >= CONSECUTIVE_LOSS_THRESHOLD:
                    logger.warning(
                        "CONSECUTIVE LOSSES: %d in a row — triggering emergency retrain",
                        self._consecutive_losses,
                    )
                    await self._retrain_worker.trigger("consecutive_losses")
                    self._consecutive_losses = 0
            else:
                self._consecutive_losses = 0

            if self.influx is not None:
                try:
                    self.influx.write_trade_outcome(data)
                except Exception as exc:
                    logger.error("influx trade outcome write failed: %s", exc)

            logger.info(
                "learned from trade %s pnl=%.4f reason=%s exchange_pnl=%s online_stats=%s",
                symbol,
                pnl,
                data.get("close_reason", ""),
                data.get("exchange_pnl", False),
                self._online_learner.stats,
            )
            if symbol:
                self._open_positions.pop(symbol, None)
            await self._retrain_worker.on_trade_closed(pnl=pnl)

        await self._pubsub_listener("execution results", self.cfg.results_channel, handle)

    async def _listen_mae_mfe(self) -> None:
        stale_cutoff = time.time() - 2100  # 35 min TTL

        async def handle(msg: dict) -> None:
            # Cleanup stale pending entries (>35 min) — add to FAISS without MAE/MFE as fallback
            stale_cutoff = time.time() - 2100
            stale_keys = [k for k, v in self._pending_mae_mfe.items() if v.get("timestamp", 0) < stale_cutoff]
            for k in stale_keys:
                stale = self._pending_mae_mfe.pop(k, None)
                if stale and stale.get("state_vector"):
                    vec = np.array(stale["state_vector"], dtype=np.float32)
                    if vec.size >= self.cfg.state_dim:
                        self.memory.add(vec[: self.cfg.state_dim], stale["pnl"], stale["regime"],
                                        stale["direction"], symbol=stale.get("symbol", ""))
                        logger.info("Added stale entry to FAISS: %s pnl=%.4f", k, stale["pnl"])

            data = json.loads(msg["data"])
            symbol = data.get("symbol", "")
            trade_id = data.get("trade_id", "")
            mae_pct = float(data.get("mae_pct", 0))
            mfe_pct = float(data.get("mfe_pct", 0))
            direction = data.get("direction", "")
            logger.info("MAE/MFE received: %s %s trade_id=%s MAE=%.2f%% MFE=%.2f%%",
                        symbol, direction, trade_id, mae_pct, mfe_pct)

            lookup_key = trade_id or symbol
            
            # Normalize lookup key — replace True/False with true/false for consistent matching
            normalized = lookup_key.replace(":True", ":true").replace(":False", ":false")
            
            # Try in-memory cache first
            pending = self._pending_mae_mfe.pop(normalized, None)
            if not pending:
                pending = self._pending_mae_mfe.pop(lookup_key, None)

            # Fallback: try Redis with multiple key formats
            if not pending:
                base = normalized.split(":")[0]  # UUID part
                for suffix in ["", ":true", ":false", ":True", ":False"]:
                    try:
                        raw = await self._redis.get(f"mae_mfe_pending:{base}{suffix}")
                        if raw:
                            pending = json.loads(raw)
                            await self._redis.delete(f"mae_mfe_pending:{base}{suffix}")
                            # Also delete other format variants
                            for alt in [":True", ":False", ":true", ":false"]:
                                if alt != suffix:
                                    await self._redis.delete(f"mae_mfe_pending:{base}{alt}")
                            break
                    except Exception:
                        pass

            if not pending:
                logger.warning("MAE/MFE for %s but no pending state_vector found", symbol)
                return

            vec = np.array(pending["state_vector"], dtype=np.float32)
            regime = pending["regime"]
            pnl = pending["pnl"]
            close_reason = pending.get("close_reason", "")

            from src.trade_judge import classify_trade
            verdict = classify_trade(
                entry_price=0, exit_price=0,
                direction=direction, pnl=pnl,
                close_reason=close_reason,
                mae_pct=mae_pct / 100.0, mfe_pct=mfe_pct / 100.0,
            )

            if vec.size >= self.cfg.state_dim:
                weight = 1.0 if pending.get("exchange_pnl", True) else 0.3
                vid = self.memory.add(vec[: self.cfg.state_dim], pnl, regime, direction,
                                      symbol=symbol, mae_pct=mae_pct / 100.0, mfe_pct=mfe_pct / 100.0,
                                      trade_judge=verdict.judge, weight=weight)
                if vid >= 0:
                    logger.info("FAISS MAE/MFE: %s vid=%d judge=%s pnl=%.4f MAE=%.2f%% MFE=%.2f%% reason=%s",
                                symbol, vid, verdict.judge, pnl, mae_pct, mfe_pct, verdict.reason)
                prom.faiss_index_size.set(self.memory.size)

        await self._pubsub_listener("mae_mfe", "execution:mae_mfe", handle)

    async def _listen_positions(self) -> None:
        async def handle(msg: dict) -> None:
            data = json.loads(msg["data"])
            if data.get("event") != "opened":
                return
            sym = data.get("symbol")
            if not sym:
                return
            self._open_positions[sym] = {
                "direction": data.get("direction", "LONG"),
                "signal_id": data.get("signal_id", ""),
                "entry_confidence": float(data.get("confidence", 0)),
                "decay_signaled": False,
                "adverse_streak": 0,
            }
            logger.info(
                "tracking open position %s %s conf=%.3f signal_id=%s",
                sym,
                self._open_positions[sym]["direction"],
                self._open_positions[sym]["entry_confidence"],
                self._open_positions[sym]["signal_id"],
            )

        await self._pubsub_listener("positions", self.cfg.positions_channel, handle)

    async def _listen_pending_orders(self) -> None:
        async def handle(msg: dict) -> None:
            data = json.loads(msg["data"])
            event = data.get("event", "")
            sym = data.get("symbol")
            if not sym:
                return
            if event in ("placed", "repriced"):
                self._pending_entries[sym] = {
                    "direction": data.get("direction", "LONG"),
                    "signal_id": data.get("signal_id", ""),
                    "entry_confidence": float(data.get("confidence", 0)),
                    "abort_signaled": False,
                    "adverse_streak": 0,
                }
                logger.info(
                    "tracking pending entry %s %s event=%s conf=%.3f signal_id=%s",
                    sym,
                    self._pending_entries[sym]["direction"],
                    event,
                    self._pending_entries[sym]["entry_confidence"],
                    self._pending_entries[sym]["signal_id"],
                )
            elif event in ("cancelled", "filled"):
                self._pending_entries.pop(sym, None)
                logger.info("pending entry cleared %s event=%s reason=%s", sym, event, data.get("reason", ""))

        await self._pubsub_listener("pending orders", self.cfg.pending_orders_channel, handle)

    async def _listen_model_reload(self) -> None:
        async def handle(_msg: dict) -> None:
            if self._hot_swap_reload():
                logger.info("model reload via redis channel")

        await self._pubsub_listener("model reload", self.cfg.model_reload_channel, handle)

    async def _model_watch_loop(self) -> None:
        while True:
            await asyncio.sleep(self.cfg.model_watch_interval_sec)
            if self._hot_swap_reload(force=False):
                logger.info("model reload via manifest/file watch")

    def _hot_swap_reload(self, force: bool = True) -> bool:
        try:
            if self.inference.reload(force=force):
                manifest = ModelManifest.from_dir(self.cfg.model_dir)
                prom.record_hot_swap_success(manifest.version, manifest.updated_at)
                return True
            return False
        except Exception as exc:
            logger.error("hot-swap failed: %s", exc)
            prom.record_hot_swap_failure()
            return False

    async def _regime_loop(self) -> None:
        while True:
            await asyncio.sleep(self.cfg.regime_interval_sec)
            for sym in list(self.active_symbols):
                buf = self.features.get(sym)
                prices = [p for _, p in buf.price_history]
                if len(prices) < 10:
                    continue
                returns = np.diff(prices) / np.array(prices[:-1])
                regime = self.regime.update(sym, returns)
                prom.regime_state.labels(symbol=sym).set(prom.REGIME_MAP.get(regime.value, 0))

    async def _persist_loop(self) -> None:
        while True:
            await asyncio.sleep(self.cfg.faiss_persist_interval_sec)
            try:
                self.memory.persist()
                self._online_learner.persist()
                logger.info("faiss index persisted, size=%d online=%s",
                            self.memory.size, self._online_learner.stats)
            except Exception as exc:
                logger.error("faiss persist failed: %s", exc)

    async def _metrics_loop(self) -> None:
        while True:
            prom.update_system_metrics()
            await asyncio.sleep(10)

    def _decode_payload(self, raw: bytes) -> dict:
        try:
            return msgpack.unpackb(raw, raw=False)
        except Exception:
            return json.loads(raw)

    async def _status_loop(self) -> None:
        interval = int(os.getenv("STATUS_LOG_INTERVAL_SEC", "60"))
        while True:
            await asyncio.sleep(interval)
            if not self.active_symbols:
                logger.info(
                    "status: waiting for screener | model=%s faiss=%d retrain_trades=%d",
                    self.inference.version,
                    self.memory.size,
                    self._retrain_worker.trades_since_retrain,
                )
                continue

            buffer_sizes = {
                sym: len(self.features.get(sym).points)
                for sym in sorted(self.active_symbols)[:8]
            }
            drought_min = (time.time() - self._last_signal_at) / 60.0
            logger.info(
                "status: symbols=%d market_events=%d predictions=%d signals=%d drought=%.0fm "
                "holds(low_conf=%d dir=%d corr=%d warming=%d pattern_blocks=%d) "
                "faiss=%d retrain_trades=%d/%d buffers=%s",
                len(self.active_symbols),
                self._stats["market_events"],
                self._stats["predictions"],
                self._stats["signals"],
                drought_min,
                self._stats["hold_low_conf"],
                self._stats["hold_direction"],
                self._stats["hold_correlation"],
                self._stats["buffer_warming"],
                self._stats["pattern_blocks"],
                self.memory.size,
                self._retrain_worker.trades_since_retrain,
                self.cfg.retrain_trade_threshold,
                buffer_sizes,
            )

            # Reset circuit breaker counters each status cycle
            with self._circuit_breaker_lock:
                self._pattern_block_count_this_cycle = 0
                self._tick_cycle_predictions = 0
                self._corr_block_count_this_cycle = 0

            SIGNAL_DROUGHT_WARN_MIN = int(os.getenv("SIGNAL_DROUGHT_WARN_MIN", "30"))
            SIGNAL_DROUGHT_RETRAIN_MIN = int(os.getenv("SIGNAL_DROUGHT_RETRAIN_MIN", "60"))
            if drought_min >= SIGNAL_DROUGHT_RETRAIN_MIN and not self._signal_drought_warned:
                logger.warning(
                    "SIGNAL DROUGHT %.0f min (threshold %d min) — triggering emergency retrain",
                    drought_min, SIGNAL_DROUGHT_RETRAIN_MIN,
                )
                self._signal_drought_warned = True
                await self._retrain_worker.trigger("signal_drought")
            elif drought_min >= SIGNAL_DROUGHT_WARN_MIN and not self._signal_drought_warned:
                logger.warning(
                    "signal drought warning: %.0f min since last signal (threshold %d min)",
                    drought_min, SIGNAL_DROUGHT_WARN_MIN,
                )

    @staticmethod
    def _payload_list(payload: dict, *keys: str) -> list:
        for key in keys:
            val = payload.get(key)
            if val:
                return val
        return []

    @staticmethod
    def _payload_scalar(payload: dict, *keys: str, default=0):
        for key in keys:
            if key in payload and payload[key] is not None:
                return payload[key]
        return default

    async def _maybe_record_telemetry(self, symbol: str, buf) -> None:
        if self.influx is None or symbol not in self.active_symbols:
            return
        now = time.time()
        last = self._last_telemetry.get(symbol, 0.0)
        if now - last < self.cfg.telemetry_interval_sec:
            return
        self._last_telemetry[symbol] = now

        state = self._event_states.get(symbol)
        snap, events = compute_telemetry_and_events(buf, state)
        mov_state = self._movement_states.get(symbol)
        mov = softmax_movement_probability(
            buf,
            mov_state,
            weights=self._movement_weights,
            bias=self._movement_bias,
            funding_prior_scale=self._movement_funding_prior,
        )
        ts_ms = int(time.time() * 1000)
        try:
            await asyncio.to_thread(
                self.influx.write_market_telemetry, symbol, snap.as_fields(), ts_ms,
            )
            await asyncio.to_thread(
                self.influx.write_movement_probability, symbol, mov.as_fields(), ts_ms,
            )
            for ev in events:
                await asyncio.to_thread(
                    self.influx.write_pattern_event,
                    symbol,
                    ev.event_type,
                    ev.price,
                    ev.score,
                    ev.detail,
                    ts_ms,
                )
                logger.info(
                    "pattern event %s %s price=%.6f score=%.2f %s",
                    symbol, ev.event_type, ev.price, ev.score, ev.detail,
                )
        except Exception as exc:
            logger.warning("telemetry write failed symbol=%s err=%s", symbol, exc)

    async def _process_market_event(self, channel: str, payload: dict, recv_at: float) -> None:
        prom.redis_pubsub_latency.observe(time.time() - recv_at)
        parts = channel.split(":")
        if len(parts) < 3:
            return
        symbol = parts[2]
        buf = self.features.get(symbol)

        self._stats["market_events"] += 1
        if "orderbook" in channel:
            self._stats["orderbook_events"] += 1
            buf.add_orderbook(
                self._payload_scalar(payload, "ts", "Ts", default=int(time.time() * 1000)),
                self._payload_list(payload, "b", "bids", "Bids"),
                self._payload_list(payload, "a", "asks", "Asks"),
            )
        elif "trades" in channel:
            self._stats["trade_events"] += 1
            buf.add_trade(
                self._payload_scalar(payload, "ts", "Ts", default=int(time.time() * 1000)),
                float(self._payload_scalar(payload, "p", "price", "Price", default=0)),
                float(self._payload_scalar(payload, "v", "size", "Size", default=0)),
                str(self._payload_scalar(payload, "S", "side", "Side", default="Buy")),
            )

        if len(buf.points) < 20:
            self._stats["buffer_warming"] += 1
            return

        await self._maybe_record_telemetry(symbol, buf)

        if symbol in self._open_positions:
            decay_payload = await asyncio.to_thread(self._check_confidence_decay, symbol, buf)
            if decay_payload is not None:
                assert self._redis is not None
                await self._redis.publish(self.cfg.signals_channel, json.dumps(decay_payload))
                prom.signals_published.inc()
                self._stats["signals"] += 1
                logger.warning(
                    "confidence decay exit %s %s reason=%s conf=%.3f kill=%.3f",
                    decay_payload["symbol"],
                    decay_payload["direction"],
                    decay_payload.get("decay_reason", "unknown"),
                    decay_payload["confidence"],
                    self.cfg.kill_confidence_threshold,
                )
            return

        if symbol in self._pending_entries:
            abort_payload = await asyncio.to_thread(self._check_entry_abort, symbol, buf)
            if abort_payload is not None:
                assert self._redis is not None
                await self._redis.publish(self.cfg.signals_channel, json.dumps(abort_payload))
                prom.signals_published.inc()
                self._stats["signals"] += 1
                logger.warning(
                    "entry abort %s %s reason=%s conf=%.3f",
                    abort_payload["symbol"],
                    abort_payload["direction"],
                    abort_payload.get("abort_reason", "confidence_decay"),
                    abort_payload["confidence"],
                )
            return

        signal_payload = await asyncio.to_thread(self._run_tick_prediction, symbol, buf)
        if signal_payload is None:
            return

        assert self._redis is not None
        await self._redis.publish(self.cfg.signals_channel, json.dumps(signal_payload))
        prom.signals_published.inc()
        self._stats["signals"] += 1
        self._last_signal_at = time.time()
        self._signal_drought_warned = False
        logger.info(
            "signal %s %s conf=%.3f vol=%.2f regime=%s sl=%.6f tps=%s trend_5m=%.4f%% trend_15m=%.4f%%",
            signal_payload["symbol"],
            signal_payload["direction"],
            signal_payload["confidence"],
            signal_payload["volatility_multiplier"],
            signal_payload["regime"],
            signal_payload.get("stop_loss", 0.0),
            signal_payload.get("tp_prices") or signal_payload.get("take_profits", []),
            signal_payload.get("macro_trend_5m", 0.0) * 100,
            signal_payload.get("macro_trend_15m", 0.0) * 100,
        )

    def _score_symbol(self, symbol: str, buf):
        """Run ONNX inference and return (direction, confidence, vol_mult, trap_prob, state_vec, regime, memory_info, v_memory).

        In PREDICT_PNL mode, confidence = normalized |pred_pnl|, trap_prob and toxic_prob are available.
        """
        flow_seq = buf.flow_sequence()
        macro = buf.feature_vector()

        ob_seq = buf.orderbook_sequence()
        v_state = self.inference.infer_state_vector(ob_seq, flow_seq, macro)
        regime = self.regime.get(symbol)
        v_memory, memory_info = self.memory.query_with_metadata(v_state, regime.value, symbol=symbol)

        predict_pnl = os.getenv("PREDICT_PNL", "false").lower() == "true"

        if predict_pnl:
            pred_pnl, trap_prob, toxic_prob = self.inference.decide_pnl(v_state, v_memory)
            direction = "LONG" if pred_pnl > 0 else "SHORT" if pred_pnl < 0 else "HOLD"
            confidence = min(abs(pred_pnl) / 0.01, 1.0) if abs(pred_pnl) >= 0.0015 else 0.0
            vol_mult = 1.0
            # Store toxic_prob for later use
            self._toxic_prob_cache = toxic_prob
            logger.info("PnL prediction: %s pred_pnl=%.4f dir=%s conf=%.3f toxic=%.3f trap=%.3f",
                        symbol, pred_pnl, direction, confidence, toxic_prob, trap_prob)
        else:
            direction, confidence, vol_mult, trap_prob = self.inference.decide(v_state, v_memory)
            self._toxic_prob_cache = 0.0

        vol_mult = min(max(vol_mult, 0.5), self.cfg.vol_multiplier_cap)
        return direction, confidence, vol_mult, trap_prob, v_state, regime, memory_info, v_memory

    def _obi_against(self, direction: str, buf) -> bool:
        obi = buf.order_book_imbalance()
        thr = self.cfg.decay_obi_threshold
        if direction == "SHORT":
            return obi > thr
        return obi < -thr

    def _trend_against(self, direction: str, buf, regime) -> bool:
        trend = buf.macro_trend(300)
        if direction == "SHORT":
            return regime.value in ("Trending", "Breakout") and trend > 0.0008
        return regime.value in ("Trending", "Breakout") and trend < -0.0008

    def _microstructure_adverse_tick(self, direction: str, buf, regime) -> bool:
        """Single orderbook event: book imbalance or trend opposes our side."""
        if self._obi_against(direction, buf):
            return True
        return self._trend_against(direction, buf, regime)

    def _adverse_events_confirmed(self, state: dict, adverse: bool) -> bool:
        """Act only after N consecutive adverse market events (orderbook updates)."""
        if not adverse:
            state["adverse_streak"] = 0
            return False
        streak = state.get("adverse_streak", 0) + 1
        state["adverse_streak"] = streak
        return streak >= self.cfg.adverse_confirm_ticks

    def _microstructure_against(self, direction: str, buf, regime) -> bool:
        return self._microstructure_adverse_tick(direction, buf, regime)

    def _build_confidence_decay_payload(
        self,
        symbol: str,
        open_pos: _OpenPosition,
        confidence: float,
        vol_mult: float,
        v_state,
        regime,
        buf,
        decay_reason: str,
    ) -> dict:
        return {
            "symbol": symbol,
            "direction": open_pos["direction"],
            "confidence": confidence,
            "volatility_multiplier": vol_mult,
            "state_vector": v_state.tolist(),
            "funding_rate": self._funding_rates.get(symbol, buf.funding_rate),
            "regime": regime.value,
            "btc_correlation": self.features.correlation_with_btc(symbol),
            "position_scale": 1.0,
            "timestamp": int(time.time() * 1000),
            "signal_id": open_pos["signal_id"] or str(uuid.uuid4()),
            "exit_reason": "confidence_decay",
            "decay_reason": decay_reason,
        }

    def _check_confidence_decay(self, symbol: str, buf) -> dict | None:
        open_pos = self._open_positions.get(symbol)
        if open_pos is None or open_pos.get("decay_signaled"):
            return None

        direction, confidence, vol_mult, trap_prob, v_state, regime, _, _ = self._score_symbol(symbol, buf)
        self._stats["predictions"] += 1
        prom.confidence_score.labels(symbol=symbol).set(confidence)

        open_dir = open_pos["direction"]

        # Confident opposite vector — require consecutive ticks to avoid flip noise.
        if direction in ("LONG", "SHORT") and direction != open_dir:
            if confidence >= self.cfg.confidence_threshold:
                streak = open_pos.get("flip_streak", 0) + 1
                open_pos["flip_streak"] = streak
                if streak >= self.cfg.direction_flip_confirm_ticks:
                    open_pos["decay_signaled"] = True
                    open_pos.pop("flip_streak", None)
                    return self._build_confidence_decay_payload(
                        symbol, open_pos, confidence, vol_mult, v_state, regime, buf, "direction_flip",
                    )
            return None

        open_pos.pop("flip_streak", None)

        # HOLD / low-confidence noise is not a reversal signal by itself.
        if direction == "HOLD":
            return None

        if confidence >= self.cfg.kill_confidence_threshold:
            open_pos["adverse_streak"] = 0
            return None

        # Microstructure adverse: if OBI and trend oppose position, force exit.
        adverse = self._microstructure_adverse_tick(open_dir, buf, regime)
        if not self._adverse_events_confirmed(open_pos, adverse):
            return None

        open_pos["decay_signaled"] = True
        return self._build_confidence_decay_payload(
            symbol, open_pos, confidence, vol_mult, v_state, regime, buf, "microstructure_adverse",
        )

    def _check_entry_abort(self, symbol: str, buf) -> dict | None:
        pending = self._pending_entries.get(symbol)
        if pending is None or pending.get("abort_signaled"):
            return None

        direction, confidence, vol_mult, trap_prob, v_state, regime, _, _ = self._score_symbol(symbol, buf)
        self._stats["predictions"] += 1
        prom.confidence_score.labels(symbol=symbol).set(confidence)

        pending_dir = pending["direction"]
        if direction in ("LONG", "SHORT") and direction != pending_dir:
            streak = pending.get("flip_streak", 0) + 1
            pending["flip_streak"] = streak
            if streak >= self.cfg.direction_flip_confirm_ticks:
                pending["abort_signaled"] = True
                pending.pop("flip_streak", None)
                return self._build_entry_abort_payload(
                    symbol, pending, confidence, vol_mult, v_state, regime, buf, "direction_flip",
                )
            return None
        pending.pop("flip_streak", None)

        # HOLD is neutral noise — keep pegging the pending limit.
        if direction == "HOLD":
            return None

        if confidence >= self.cfg.entry_abort_threshold:
            pending["adverse_streak"] = 0
            return None

        # Microstructure adverse is too aggressive for pending entries — only direction flip matters.
        return None

    def _build_entry_abort_payload(
        self,
        symbol: str,
        pending: _PendingEntry,
        confidence: float,
        vol_mult: float,
        v_state,
        regime,
        buf,
        abort_reason: str,
    ) -> dict:
        return {
            "symbol": symbol,
            "direction": pending["direction"],
            "confidence": confidence,
            "volatility_multiplier": vol_mult,
            "state_vector": v_state.tolist(),
            "funding_rate": self._funding_rates.get(symbol, buf.funding_rate),
            "regime": regime.value,
            "btc_correlation": self.features.correlation_with_btc(symbol),
            "position_scale": 1.0,
            "timestamp": int(time.time() * 1000),
            "signal_id": pending["signal_id"] or str(uuid.uuid4()),
            "setup_action": "abort_setup",
            "abort_reason": abort_reason,
        }

    def _run_tick_prediction(self, symbol: str, buf) -> dict | None:
        """CPU-bound ONNX inference — runs in a thread pool to keep the event loop responsive."""
        tick_start = time.time()
        direction, confidence, vol_mult, trap_prob, v_state, regime, mem_info, v_memory = self._score_symbol(symbol, buf)
        self._stats["predictions"] += 1
        self._tick_cycle_predictions += 1
        corr = self.features.correlation_with_btc(symbol)
        prom.btc_correlation.labels(symbol=symbol).set(corr)

        # Trap Head: now in LOGGING mode — reduce position size instead of blocking
        TRAP_THRESHOLD = float(os.getenv("TRAP_THRESHOLD", "0.60"))
        if trap_prob > TRAP_THRESHOLD:
            self._stats["trap_blocked"] = self._stats.get("trap_blocked", 0) + 1
            if confidence > 0.85:
                vol_mult *= 0.5
                logger.info("Trap Head PARTIAL for %s: trap=%.3f conf=%.3f → vol*=0.5",
                            symbol, trap_prob, confidence)
            else:
                logger.info("Trap Head LOG: %s trap=%.3f dir=%s conf=%.3f (not blocking)",
                            symbol, trap_prob, direction, confidence)

        predict_pnl = os.getenv("PREDICT_PNL", "false").lower() == "true"
        if predict_pnl:
            toxic_prob = self._toxic_prob_cache
            TOXIC_THRESHOLD = float(os.getenv("TOXIC_THRESHOLD", "0.35"))
            if toxic_prob > TOXIC_THRESHOLD:
                self._stats["toxic_blocked"] = self._stats.get("toxic_blocked", 0) + 1
                logger.info("TOXIC FLOW BLOCK: %s toxic=%.3f > %.3f → HOLD",
                            symbol, toxic_prob, TOXIC_THRESHOLD)
                return None
            min_pnl = float(os.getenv("MIN_PNL_THRESHOLD", "0.0015"))
            if confidence < min_pnl / 0.01:
                self._stats["hold_low_conf"] += 1
                return None

        position_scale = 1.0
        CORRELATION_BLOCK_LIMIT = int(os.getenv("CORRELATION_BLOCK_LIMIT", "8"))
        corr_blocked = False
        if corr > self.cfg.correlation_threshold and symbol != "BTCUSDT":
            with self._circuit_breaker_lock:
                corr_limit_reached = self._corr_block_count_this_cycle >= CORRELATION_BLOCK_LIMIT
            if confidence < 0.95 and not corr_limit_reached:
                direction = "HOLD"
                self._stats["hold_correlation"] += 1
                corr_blocked = True
                with self._circuit_breaker_lock:
                    self._corr_block_count_this_cycle += 1
            position_scale = 0.25

        # Circuit breaker: stop EV gate / pattern blocks if too many symbols already blocked this cycle
        # Prevents death spiral where all signals get filtered
        PATTERN_BLOCK_LIMIT = int(os.getenv("PATTERN_BLOCK_LIMIT", "50"))
        with self._circuit_breaker_lock:
            cycle_limit_reached = self._pattern_block_count_this_cycle >= PATTERN_BLOCK_LIMIT

        # Direction gate: DISABLED while data insufficient
        # When FAISS has 500+ entries, re-enable with relaxed thresholds

        # Pattern Memory: block trades with similar losing patterns
        if direction != "HOLD" and not cycle_limit_reached:
            should_avoid, avg_loss = self._online_learner.pattern_memory.should_avoid(
                v_state, regime.value, direction, symbol=symbol,
            )
            if should_avoid:
                self._stats["pattern_blocks"] += 1
                self._symbol_cooldowns[symbol] = time.time() + 1800
                logger.warning(
                    "PATTERN MEMORY BLOCK: %s %s for %s (avg_loss=%.4f)",
                    direction, symbol, regime.value, avg_loss,
                )
                return None

        # Symbol-level cooldown: block re-entry for 5 min after SL
        if symbol in self._symbol_cooldowns:
            if time.time() < self._symbol_cooldowns[symbol]:
                self._stats["hold_low_conf"] += 1
                return None
            else:
                del self._symbol_cooldowns[symbol]

        prom.confidence_score.labels(symbol=symbol).set(confidence)
        prom.volatility_multiplier.labels(symbol=symbol).set(vol_mult)
        prom.tick_to_signal_latency.observe(time.time() - tick_start)

        if confidence > self.cfg.max_signal_confidence:
            logger.warning("Signal anomaly filtered for symbol %s conf=%.3f", symbol, confidence)
            return None

        if direction == "HOLD":
            self._stats["hold_direction"] += 1
            return None

        # Direction-specific thresholds: LONG needs higher bar (model is biased toward LONG)
        long_conf_threshold = self.cfg.long_confidence_threshold
        trend_5m = buf.macro_trend(300)
        trend_15m = buf.macro_trend(900)
        bullish = trend_5m > self.cfg.bullish_trend_threshold and trend_15m > 0
        shadow_only = False  # Flag: signal only for shadow training, no real trade

        # Dynamic per-symbol confidence: raise threshold for symbols with poor WR
        base_threshold = self.cfg.confidence_threshold
        st = self._symbol_trades.get(symbol, {})
        total = st.get("wins", 0) + st.get("losses", 0)
        if total >= 3:
            wr = st.get("wins", 0) / total
            if wr < 0.40:
                penalty = max(1.0, 3.0 - 2.5 * (wr / 0.40))
                base_threshold = min(self.cfg.confidence_threshold * penalty, 0.95)
                logger.info("Dynamic conf: %s WR=%.0f%% (%d/%d) → threshold=%.3f (base=%.3f)",
                           symbol, wr * 100, st.get("wins", 0), total, base_threshold, self.cfg.confidence_threshold)

        # Symbol+Setup escalation: 3+ losses in same symbol+regime+direction → threshold 0.80
        setup_key = f"{symbol}_{direction}_{regime}"
        setup_losses = self._symbol_setup_losses.get(setup_key, [])
        if len(setup_losses) >= 3:
            avg_setup_loss = sum(setup_losses) / len(setup_losses)
            if avg_setup_loss < -0.10:
                escalation = 0.80
                if escalation > base_threshold:
                    base_threshold = escalation
                    logger.info("Symbol+Setup escalation: %s %s %s avg_loss=%.4f (%d losses) → threshold=%.3f",
                               symbol, direction, regime, avg_setup_loss, len(setup_losses), base_threshold)

        # Hard trend filter: block SHORT in uptrend, LONG in downtrend
        # Counter-trend flip: if trend opposes, flip direction
        flipped = False
        if direction == "SHORT" and trend_5m > 0.003:
            # Strong uptrend → flip to LONG for real trade
            logger.info("Trend flip: %s SHORT→LONG (trend_5m=%.4f%% > 0.3%%)",
                       symbol, trend_5m * 100)
            direction = "LONG"
            flipped = True
        elif direction == "SHORT" and trend_5m > 0:
            # Mild uptrend → block SHORT (don't trade against trend)
            self._stats["hold_direction"] += 1
            logger.info("Trend filter: %s SHORT blocked (trend_5m=%.4f%% > 0)",
                       symbol, trend_5m * 100)
            return None
        elif direction == "LONG" and trend_5m < -0.003:
            # Strong downtrend → flip to SHORT for real, but allow shadow LONG
            logger.info("Trend flip: %s LONG→SHORT (trend_5m=%.4f%% < -0.3%%)",
                       symbol, trend_5m * 100)
            direction = "SHORT"
        elif direction == "LONG" and trend_5m < 0:
            logger.info("Trend filter: %s LONG in downtrend (trend_5m=%.4f%% < 0, conf=%.3f)",
                       symbol, trend_5m * 100, confidence)

        # Only relax when: bullish trend + model shows positive long EV + confidence >= 0.50
        long_ev = mem_info.get("long_avg_pnl", 0.0)
        long_matches = mem_info.get("long_matches", 0)
        if (bullish and direction == "LONG" and long_ev > 0 and long_matches >= 3
                and confidence >= 0.50):
            long_conf_threshold = self.cfg.bullish_long_conf_threshold
            logger.info("Bullish trend + positive long EV for %s: trend_5m=%.4f long_ev=$%.4f conf=%.3f, threshold lowered to %.3f",
                       symbol, trend_5m, long_ev, confidence, long_conf_threshold)

        if direction == "SHORT":
            threshold = base_threshold
        elif flipped:
            threshold = base_threshold
        else:
            threshold = long_conf_threshold

        if confidence <= threshold:
            if shadow_only:
                logger.info("Shadow-only LONG: %s conf=%.3f below threshold %.3f, keeping for shadow",
                           symbol, confidence, threshold)
            else:
                self._stats["hold_low_conf"] += 1
                return None

        # LONG trend filter: allow LONG even in downtrend with reduced confidence
        if direction == "LONG":
            trend_5m = buf.macro_trend(300)
            trend_15m = buf.macro_trend(900)
            if trend_5m < 0 and trend_15m < 0:
                confidence = min(confidence, 0.30)
                logger.info("LONG trend filter: %s trend_5m=%.4f trend_15m=%.4f → reduced conf=%.3f",
                            symbol, trend_5m, trend_15m, confidence)

        dynamic_sl = mem_info.get("dynamic_sl_pct", 0.0)
        dynamic_tp = mem_info.get("dynamic_tp_pct", 0.0)
        salvageable = mem_info.get("salvageable_count", 0)
        unsalvageable = mem_info.get("unsalvageable_count", 0)
        salvageable_ratio = salvageable / max(salvageable + unsalvageable, 1)

        exit_sl, exit_tp, trade_score = self.inference.infer_exit_params(
            v_state, v_memory[:8] if len(v_memory) >= 8 else np.zeros(8),
            direction, regime.value, confidence, vol_mult,
            dynamic_sl, dynamic_tp, salvageable_ratio,
        )

        exit_sl = max(min(exit_sl, self.cfg.max_sl_pct), self.cfg.min_sl_pct)
        exit_tp = max(min(exit_tp, self.cfg.max_tp_pct), self.cfg.min_tp_pct)

        if dynamic_sl > 0 and dynamic_sl < exit_sl and dynamic_sl >= self.cfg.min_sl_pct:
            exit_sl = dynamic_sl
        if dynamic_tp > 0 and dynamic_tp < exit_tp and dynamic_tp >= self.cfg.min_tp_pct:
            exit_tp = dynamic_tp

        logger.info("ExitOptimizer: %s sl=%.3f%% tp=%.3f%% score=%.3f (regime=%s conf=%.3f)",
                    symbol, exit_sl * 100, exit_tp * 100, trade_score, regime.value, confidence)

        exit_plan = build_exit_plan(
            direction,
            buf.latest_bids,
            buf.latest_asks,
            vol_mult,
            regime.value,
            confidence,
            fallback_mid=buf.last_mid(),
            vol_multiplier_cap=self.cfg.vol_multiplier_cap,
            entry_maker_ticks=self.cfg.entry_maker_ticks,
            min_tp_pct=exit_tp * 0.8,
            fee_breakeven_pct=self.cfg.fee_breakeven_pct,
            max_tp_pct=exit_tp,
            max_sl_pct=exit_sl,
            macro_trend_5m=buf.macro_trend(300),
            macro_trend_15m=buf.macro_trend(900),
        )

        signal = {
            "symbol": symbol,
            "direction": direction,
            "confidence": confidence,
            "volatility_multiplier": vol_mult,
            "state_vector": v_state.tolist(),
            "funding_rate": self._funding_rates.get(symbol, buf.funding_rate),
            "regime": regime.value,
            "btc_correlation": corr,
            "position_scale": position_scale,
            "macro_trend_5m": buf.macro_trend(300),
            "macro_trend_15m": buf.macro_trend(900),
            "timestamp": int(time.time() * 1000),
            "signal_id": str(uuid.uuid4()),
        }

        dynamic_sl = mem_info.get("dynamic_sl_pct", 0.0)
        dynamic_tp = mem_info.get("dynamic_tp_pct", 0.0)
        salvageable = mem_info.get("salvageable_count", 0)
        unsalvageable = mem_info.get("unsalvage_count", 0)

        if exit_sl > 0 and exit_tp > 0:
            signal["dynamic_sl_pct"] = exit_sl
            signal["dynamic_tp_pct"] = exit_tp
            logger.info("ExitOptimizer: %s SL=%.3f%% TP=%.3f%% score=%.3f",
                        symbol, exit_sl * 100, exit_tp * 100, trade_score)
        elif dynamic_sl > 0:
            signal["dynamic_sl_pct"] = dynamic_sl
            signal["dynamic_tp_pct"] = dynamic_tp
            logger.info("Parametric Memory: %s SL=%.3f%% TP=%.3f%% (salv=%d unsalv=%d)",
                        symbol, dynamic_sl * 100, dynamic_tp * 100, salvageable, unsalvageable)

        if exit_plan:
            signal["entry_price"] = exit_plan["entry_price"]
            signal["stop_loss"] = exit_plan["stop_loss"]
            signal["take_profits"] = exit_plan["take_profits"]
            if exit_plan.get("tp_prices"):
                signal["tp_prices"] = exit_plan["tp_prices"]
            signal["wall_price"] = exit_plan.get("wall_price", 0.0)
            trend_penalty = exit_plan.get("trend_penalty", 0.0)
            if trend_penalty > 0:
                trade_score = max(0.0, trade_score - trend_penalty)
                signal["trade_score"] = trade_score
                logger.info("Trend penalty applied: %s penalty=%.3f adjusted_score=%.3f",
                           symbol, trend_penalty, trade_score)

        signal["exit_opt_sl_pct"] = exit_sl
        signal["exit_opt_tp_pct"] = exit_tp
        signal["exit_trade_score"] = trade_score
        signal["shadow_only"] = shadow_only

        if trade_score < 0.3:
            self._stats["hold_low_conf"] += 1
            logger.info("ExitOptimizer LOW SCORE: %s score=%.3f sl=%.3f%% tp=%.3f%% (signal blocked)",
                        symbol, trade_score, exit_sl * 100, exit_tp * 100)
            return None

        return signal
