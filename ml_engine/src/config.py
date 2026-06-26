"""Application configuration loaded from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    redis_addr: str
    redis_host: str
    redis_port: int
    active_symbols_channel: str
    signals_channel: str
    results_channel: str
    positions_channel: str
    pending_orders_channel: str
    model_reload_channel: str
    metrics_addr: str
    model_dir: str
    faiss_path: str
    confidence_threshold: float
    kill_confidence_threshold: float
    entry_abort_threshold: float
    adverse_confirm_ticks: int
    direction_flip_confirm_ticks: int
    decay_obi_threshold: float
    max_signal_confidence: float
    blacklist_symbols: frozenset[str]
    vol_multiplier_cap: float
    min_tp_pct: float
    max_tp_pct: float
    fee_breakeven_pct: float
    max_sl_pct: float
    entry_maker_ticks: int
    buffer_seconds: int
    regime_interval_sec: int
    correlation_threshold: float
    memory_decay_days: int
    faiss_persist_interval_sec: int
    state_dim: int
    memory_dim: int
    influx_url: str
    influx_token: str
    influx_org: str
    influx_bucket_raw: str
    influx_bucket_features: str
    model_watch_interval_sec: int
    retrain_interval_sec: int
    retrain_trade_threshold: int
    retrain_lookback_hours: int
    retrain_epochs: int
    retrain_timeout_sec: int
    train_nice_level: int
    loss_penalty_weight: float
    train_device: str
    telemetry_interval_sec: int
    bullish_trend_threshold: float
    bullish_long_conf_threshold: float
    long_confidence_threshold: float

    @classmethod
    def load(cls) -> "Config":
        addr = os.getenv("REDIS_ADDR", "redis:6379")
        host, _, port_str = addr.partition(":")
        port = int(port_str or "6379")
        retrain_hours = float(os.getenv("RETRAIN_INTERVAL_HOURS", "2"))
        return cls(
            redis_addr=addr,
            redis_host=host or "redis",
            redis_port=port,
            active_symbols_channel=os.getenv("ACTIVE_SYMBOLS_CHANNEL", "config:active_symbols"),
            signals_channel=os.getenv("SIGNALS_CHANNEL", "orders:signals"),
            results_channel=os.getenv("RESULTS_CHANNEL", "execution:results"),
            positions_channel=os.getenv("POSITIONS_CHANNEL", "execution:positions"),
            pending_orders_channel=os.getenv("PENDING_ORDERS_CHANNEL", "execution:pending_orders"),
            model_reload_channel=os.getenv("MODEL_RELOAD_CHANNEL", "models:reload"),
            metrics_addr=os.getenv("METRICS_ADDR", ":9103"),
            model_dir=os.getenv("MODEL_DIR", "/app/models"),
            faiss_path=os.getenv("FAISS_PATH", "/app/data/faiss_index"),
            confidence_threshold=float(os.getenv("CONFIDENCE_THRESHOLD", "0.30")),
            kill_confidence_threshold=float(os.getenv("KILL_CONFIDENCE_THRESHOLD", "0.35")),
            entry_abort_threshold=float(os.getenv("ENTRY_ABORT_THRESHOLD", "0.50")),
            adverse_confirm_ticks=int(os.getenv("ADVERSE_CONFIRM_TICKS", os.getenv("DECAY_OBI_MIN_TICKS", "3"))),
            direction_flip_confirm_ticks=int(os.getenv("DIRECTION_FLIP_CONFIRM_TICKS", "5")),
            decay_obi_threshold=float(os.getenv("DECAY_OBI_THRESHOLD", "0.30")),
            max_signal_confidence=float(os.getenv("MAX_SIGNAL_CONFIDENCE", "0.98")),
            blacklist_symbols=cls._parse_symbol_set(os.getenv("BLACKLIST_SYMBOLS", "")),
            vol_multiplier_cap=float(os.getenv("VOL_MULTIPLIER_CAP", "2.0")),
            min_tp_pct=float(os.getenv("MIN_TP_PCT", "0.002")),
            max_tp_pct=float(os.getenv("MAX_TP_PCT", "0.008")),
            fee_breakeven_pct=float(os.getenv("FEE_BREAKEVEN_PCT", "0.0015")),
            max_sl_pct=float(os.getenv("MAX_SL_PCT", "0.012")),
            entry_maker_ticks=int(os.getenv("ENTRY_MAKER_TICKS", "2")),
            buffer_seconds=int(os.getenv("BUFFER_SECONDS", "300")),
            regime_interval_sec=int(os.getenv("REGIME_INTERVAL_SEC", "60")),
            correlation_threshold=float(os.getenv("CORRELATION_THRESHOLD", "0.85")),
            bullish_trend_threshold=float(os.getenv("BULLISH_TREND_THRESHOLD", "0.0005")),
            bullish_long_conf_threshold=float(os.getenv("BULLISH_LONG_CONF_THRESHOLD", "0.20")),
            long_confidence_threshold=float(os.getenv("LONG_CONFIDENCE_THRESHOLD", "0.50")),
            memory_decay_days=int(os.getenv("MEMORY_DECAY_DAYS", "14")),
            faiss_persist_interval_sec=int(os.getenv("FAISS_PERSIST_INTERVAL_SEC", "300")),
            state_dim=int(os.getenv("STATE_DIM", "128")),
            memory_dim=int(os.getenv("MEMORY_DIM", "8")),
            influx_url=os.getenv("INFLUX_URL", "http://influxdb:8086"),
            influx_token=os.getenv("INFLUX_TOKEN", ""),
            influx_org=os.getenv("INFLUX_ORG", "fasttrader"),
            influx_bucket_raw=os.getenv("INFLUX_BUCKET_RAW", "market_raw"),
            influx_bucket_features=os.getenv("INFLUX_BUCKET_FEATURES", "market_features"),
            model_watch_interval_sec=int(os.getenv("MODEL_WATCH_INTERVAL_SEC", "10")),
            retrain_interval_sec=int(retrain_hours * 3600),
            retrain_trade_threshold=int(os.getenv("RETRAIN_TRADE_THRESHOLD", "200")),
            retrain_lookback_hours=int(os.getenv("RETRAIN_LOOKBACK_HOURS", "48")),
            retrain_epochs=int(os.getenv("RETRAIN_EPOCHS", "8")),
            retrain_timeout_sec=int(os.getenv("RETRAIN_TIMEOUT_SEC", "3600")),
            train_nice_level=int(os.getenv("TRAIN_NICE_LEVEL", "15")),
            loss_penalty_weight=float(os.getenv("LOSS_PENALTY_WEIGHT", "2.5")),
            train_device=os.getenv("TRAIN_DEVICE", "cuda"),
            telemetry_interval_sec=int(os.getenv("TELEMETRY_INTERVAL_SEC", "5")),
        )

    @staticmethod
    def _parse_symbol_set(raw: str) -> frozenset[str]:
        return frozenset(part.strip().upper() for part in raw.split(",") if part.strip())
