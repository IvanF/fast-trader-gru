"""InfluxDB writer for trade outcomes and training-data join helpers."""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from influxdb_client import InfluxDBClient, Point, WritePrecision
from influxdb_client.client.write_api import ASYNCHRONOUS

logger = logging.getLogger(__name__)

TRADE_MEASUREMENT = "trade_outcomes"
TELEMETRY_MEASUREMENT = "market_microstructure"
EVENT_MEASUREMENT = "pattern_events"
MOVEMENT_MEASUREMENT = "movement_probability"


class InfluxStore:
    def __init__(
        self,
        url: str,
        token: str,
        org: str,
        bucket_raw: str,
        bucket_features: str = "market_features",
    ) -> None:
        self.url = url
        self.token = token
        self.org = org
        self.bucket_raw = bucket_raw
        self.bucket_features = bucket_features
        self._client: Optional[InfluxDBClient] = None
        self._write_api = None

    def connect(self) -> None:
        if self._client is not None:
            return
        self._client = InfluxDBClient(url=self.url, token=self.token, org=self.org)
        self._write_api = self._client.write_api(write_options=ASYNCHRONOUS)
        logger.info("influx store connected url=%s bucket=%s", self.url, self.bucket_raw)

    def close(self) -> None:
        if self._write_api is not None:
            self._write_api.close()
        if self._client is not None:
            self._client.close()
        self._client = None
        self._write_api = None

    def write_trade_outcome(self, data: dict[str, Any]) -> None:
        if self._write_api is None:
            self.connect()
        assert self._write_api is not None

        closed_at = int(data.get("closed_at", time.time() * 1000))
        ts = datetime.fromtimestamp(closed_at / 1000.0, tz=timezone.utc)
        state_vec = data.get("state_vector", [])
        state_json = json.dumps(state_vec[:128]) if state_vec else "[]"

        point = (
            Point(TRADE_MEASUREMENT)
            .tag("symbol", str(data.get("symbol", "UNKNOWN")))
            .tag("direction", str(data.get("direction", "HOLD")))
            .tag("regime", str(data.get("regime", "Choppy")))
            .tag("signal_id", str(data.get("signal_id", "")))
            .tag("close_reason", str(data.get("close_reason", "")))
            .field("net_pnl", float(data.get("net_pnl", 0)))
            .field("entry_price", float(data.get("entry_price", 0)))
            .field("exit_price", float(data.get("exit_price", 0)))
            .field("holding_time_ms", int(data.get("holding_time_ms", 0)))
            .field("partial_closed", bool(data.get("partial_closed", False)))
            .field("grid_levels", int(data.get("grid_levels", 0)))
            .field("state_vector_json", state_json)
            .field("won", float(data.get("net_pnl", 0)) >= 0)
            .field("exchange_pnl", bool(data.get("exchange_pnl", False)))
            .time(ts, WritePrecision.MS)
        )
        self._write_api.write(bucket=self.bucket_raw, org=self.org, record=point)

    def write_market_telemetry(self, symbol: str, fields: dict[str, float], ts_ms: Optional[int] = None) -> None:
        if self._write_api is None:
            self.connect()
        assert self._write_api is not None
        ts = datetime.fromtimestamp((ts_ms or int(time.time() * 1000)) / 1000.0, tz=timezone.utc)
        point = Point(TELEMETRY_MEASUREMENT).tag("symbol", symbol)
        for key, val in fields.items():
            if val is None or (isinstance(val, float) and not (val == val)):  # NaN
                continue
            point = point.field(key, float(val))
        point = point.time(ts, WritePrecision.MS)
        self._write_api.write(bucket=self.bucket_features, org=self.org, record=point)

    def write_pattern_event(
        self,
        symbol: str,
        event_type: str,
        price: float,
        score: float,
        detail: str = "",
        ts_ms: Optional[int] = None,
    ) -> None:
        if self._write_api is None:
            self.connect()
        assert self._write_api is not None
        ts = datetime.fromtimestamp((ts_ms or int(time.time() * 1000)) / 1000.0, tz=timezone.utc)
        point = (
            Point(EVENT_MEASUREMENT)
            .tag("symbol", symbol)
            .tag("event_type", event_type)
            .field("price", float(price))
            .field("score", float(score))
            .field("detail", str(detail)[:256])
            .time(ts, WritePrecision.MS)
        )
        self._write_api.write(bucket=self.bucket_features, org=self.org, record=point)

    def write_movement_probability(
        self, symbol: str, fields: dict[str, float], ts_ms: Optional[int] = None
    ) -> None:
        """P(Short|X), P(Neutral|X), P(Long|X) and state vector x1..x5."""
        if self._write_api is None:
            self.connect()
        assert self._write_api is not None
        ts = datetime.fromtimestamp((ts_ms or int(time.time() * 1000)) / 1000.0, tz=timezone.utc)
        point = Point(MOVEMENT_MEASUREMENT).tag("symbol", symbol)
        for key, val in fields.items():
            if val is None or (isinstance(val, float) and not (val == val)):
                continue
            point = point.field(key, float(val))
        point = point.time(ts, WritePrecision.MS)
        self._write_api.write(bucket=self.bucket_features, org=self.org, record=point)

    def query_trade_outcomes(
        self, start: str, stop: str = "now()", symbol: Optional[str] = None
    ) -> list[dict[str, Any]]:
        if self._client is None:
            self.connect()
        assert self._client is not None

        sym_filter = ""
        if symbol:
            sym_filter = f'|> filter(fn: (r) => r.symbol == "{symbol}")'

        flux = f'''
from(bucket: "{self.bucket_raw}")
  |> range(start: {start}, stop: {stop})
  |> filter(fn: (r) => r._measurement == "{TRADE_MEASUREMENT}")
  {sym_filter}
  |> pivot(rowKey: ["_time", "symbol", "direction", "regime", "signal_id"],
           columnKey: ["_field"], valueColumn: "_value")
'''
        rows: list[dict[str, Any]] = []
        tables = self._client.query_api().query(flux, org=self.org)
        for table in tables:
            for record in table.records:
                row: dict[str, Any] = {
                    "_time": record.get_time(),
                    "symbol": record.values.get("symbol"),
                    "direction": record.values.get("direction"),
                    "regime": record.values.get("regime"),
                    "signal_id": record.values.get("signal_id"),
                }
                for key in (
                    "net_pnl", "entry_price", "exit_price", "holding_time_ms",
                    "partial_closed", "grid_levels", "state_vector_json", "won",
                ):
                    if key in record.values:
                        row[key] = record.values[key]
                rows.append(row)
        return rows

    def query_market_features_window(
        self,
        symbol: str,
        center_ts: datetime,
        window_sec: int = 300,
        bucket: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        if self._client is None:
            self.connect()
        assert self._client is not None

        start_dt = center_ts - timedelta(seconds=window_sec)
        start_iso = start_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        stop_dt = center_ts.timestamp() + 1
        stop_iso = datetime.fromtimestamp(stop_dt, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        bucket = bucket or self.bucket_features

        flux = f'''
from(bucket: "{bucket}")
  |> range(start: time(v: "{start_iso}"), stop: time(v: "{stop_iso}"))
  |> filter(fn: (r) => r._measurement == "orderbook_summary" or r._measurement == "trades")
  |> filter(fn: (r) => r.symbol == "{symbol}")
  |> pivot(rowKey: ["_time"], columnKey: ["_field"], valueColumn: "_value")
'''
        rows: list[dict[str, Any]] = []
        tables = self._client.query_api().query(flux, org=self.org)
        for table in tables:
            for record in table.records:
                row = {"_time": record.get_time()}
                for k, v in record.values.items():
                    if not k.startswith("_") or k == "_time":
                        row[k] = v
                rows.append(row)
        rows.sort(key=lambda r: r["_time"])
        return rows
