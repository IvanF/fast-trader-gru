"""Async ML engine core — Redis listener, inference loop, continuous learning."""

from __future__ import annotations

import asyncio
import json
import logging
import os
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
        }
        self._last_status_log = time.time()
        self._retrain_worker = RollingRetrainWorker(
            cfg, symbols_provider=lambda: sorted(self.active_symbols),
        )
        self._open_positions: Dict[str, _OpenPosition] = {}
        self._pending_entries: Dict[str, _PendingEntry] = {}
        self._event_states = EventStateStore()
        self._movement_states = MovementStateStore()
        self._movement_weights, self._movement_bias = load_weights_from_env()
        self._movement_funding_prior = float(os.getenv("MOVEMENT_PROB_FUNDING_PRIOR_SCALE", "50"))
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
                        continue
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

        port = int(self.cfg.metrics_addr.split(":")[-1])
        prom.start_metrics_server(port)
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
            data = json.loads(msg["data"])
            vec = np.array(data.get("state_vector", []), dtype=np.float32)
            if vec.size == 0:
                return
            if vec.size < self.cfg.state_dim:
                vec = np.pad(vec, (0, self.cfg.state_dim - vec.size))
            pnl = float(data.get("net_pnl", 0))
            regime = data.get("regime", "Choppy")
            direction = data.get("direction", "HOLD")
            self.memory.add(vec[: self.cfg.state_dim], pnl, regime, direction)
            prom.faiss_index_size.set(self.memory.size)

            if self.influx is not None:
                try:
                    self.influx.write_trade_outcome(data)
                except Exception as exc:
                    logger.error("influx trade outcome write failed: %s", exc)

            logger.info(
                "learned from trade %s pnl=%.4f reason=%s exchange_pnl=%s",
                data.get("symbol"),
                pnl,
                data.get("close_reason", ""),
                data.get("exchange_pnl", False),
            )
            sym = data.get("symbol")
            if sym:
                self._open_positions.pop(sym, None)
            await self._retrain_worker.on_trade_closed()

        await self._pubsub_listener("execution results", self.cfg.results_channel, handle)

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
                logger.info("faiss index persisted, size=%d", self.memory.size)
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
            logger.info(
                "status: symbols=%d market_events=%d predictions=%d signals=%d "
                "holds(low_conf=%d dir=%d corr=%d warming=%d) "
                "faiss=%d retrain_trades=%d/%d buffers=%s",
                len(self.active_symbols),
                self._stats["market_events"],
                self._stats["predictions"],
                self._stats["signals"],
                self._stats["hold_low_conf"],
                self._stats["hold_direction"],
                self._stats["hold_correlation"],
                self._stats["buffer_warming"],
                self.memory.size,
                self._retrain_worker.trades_since_retrain,
                self.cfg.retrain_trade_threshold,
                buffer_sizes,
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
        logger.info(
            "signal %s %s conf=%.3f vol=%.2f regime=%s sl=%.6f tps=%s",
            signal_payload["symbol"],
            signal_payload["direction"],
            signal_payload["confidence"],
            signal_payload["volatility_multiplier"],
            signal_payload["regime"],
            signal_payload.get("stop_loss", 0.0),
            signal_payload.get("tp_prices") or signal_payload.get("take_profits", []),
        )

    def _score_symbol(self, symbol: str, buf):
        """Run ONNX inference and return (direction, confidence, vol_mult, state_vec, regime)."""
        ob_seq = buf.orderbook_sequence()
        flow_seq = buf.flow_sequence()
        macro = buf.feature_vector()

        v_state = self.inference.infer_state_vector(ob_seq, flow_seq, macro)
        regime = self.regime.get(symbol)
        v_memory, _ = self.memory.query(v_state, regime.value)

        direction, confidence, vol_mult = self.inference.decide(v_state, v_memory)
        vol_mult = min(max(vol_mult, 0.5), self.cfg.vol_multiplier_cap)
        return direction, confidence, vol_mult, v_state, regime

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

        direction, confidence, vol_mult, v_state, regime = self._score_symbol(symbol, buf)
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

        direction, confidence, vol_mult, v_state, regime = self._score_symbol(symbol, buf)
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
        direction, confidence, vol_mult, v_state, regime = self._score_symbol(symbol, buf)
        self._stats["predictions"] += 1
        corr = self.features.correlation_with_btc(symbol)
        prom.btc_correlation.labels(symbol=symbol).set(corr)

        position_scale = 1.0
        if corr > self.cfg.correlation_threshold and symbol != "BTCUSDT":
            if confidence < 0.95:
                direction = "HOLD"
                self._stats["hold_correlation"] += 1
            position_scale = 0.25

        prom.confidence_score.labels(symbol=symbol).set(confidence)
        prom.volatility_multiplier.labels(symbol=symbol).set(vol_mult)
        prom.tick_to_signal_latency.observe(time.time() - tick_start)

        if confidence > self.cfg.max_signal_confidence:
            logger.warning("Signal anomaly filtered for symbol %s conf=%.3f", symbol, confidence)
            return None

        if direction == "HOLD":
            self._stats["hold_direction"] += 1
            return None

        if confidence <= self.cfg.confidence_threshold:
            self._stats["hold_low_conf"] += 1
            return None

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
            min_tp_pct=self.cfg.min_tp_pct,
            fee_breakeven_pct=self.cfg.fee_breakeven_pct,
            max_tp_pct=self.cfg.max_tp_pct,
            max_sl_pct=self.cfg.max_sl_pct,
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
        if exit_plan:
            signal["entry_price"] = exit_plan["entry_price"]
            signal["stop_loss"] = exit_plan["stop_loss"]
            signal["take_profits"] = exit_plan["take_profits"]
            if exit_plan.get("tp_prices"):
                signal["tp_prices"] = exit_plan["tp_prices"]
            signal["wall_price"] = exit_plan.get("wall_price", 0.0)
        return signal
