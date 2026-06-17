#!/usr/bin/env python3
"""
Flush dirty trade-feedback data from tonight's run (Influx trade_outcomes, FAISS,
retrain counters) while keeping DB/bucket structure intact.

Usage (recommended — uses compose network + ml_engine deps):
  docker compose run --rm --no-deps \\
    -v "$(pwd)":/workspace:ro \\
    --entrypoint python3 \\
    ml_engine /workspace/scripts/flush_dirty_data.py --env /workspace/.env --yes

From host (maps FTG_INFLUX_PORT / FTG_REDIS_PORT; pip install influxdb-client redis):
  FTG_FLUSH_USE_LOCALHOST=1 python3 scripts/flush_dirty_data.py --yes

Dry run:
  python3 scripts/flush_dirty_data.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_SINCE = "2026-06-16T18:00:00Z"
DEFAULT_INFLUX_TIMEOUT_SEC = 300
TRADE_MEASUREMENT = "trade_outcomes"
ML_CONTAINER = "ftg-ml-engine"

# Redis keys/patterns tied to live execution cache (not screener config).
REDIS_DELETE_PATTERNS = (
    "cache:orderbook:*",
)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def parse_env_file(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if not path.is_file():
        return out
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        out[key] = val
    return out


def inside_compose_network() -> bool:
    if os.getenv("FTG_FLUSH_USE_LOCALHOST", "").lower() in ("1", "true", "yes"):
        return False
    if Path("/app/data").is_dir():
        return True
    try:
        socket.gethostbyname("influxdb")
        return True
    except OSError:
        return False


def merged_config(env_path: Path) -> dict[str, str]:
    file_env = parse_env_file(env_path)
    cfg: dict[str, str] = {**file_env, **{k: v for k, v in os.environ.items() if v}}

    influx_port = cfg.get("FTG_INFLUX_PORT", "18086")
    redis_port = cfg.get("FTG_REDIS_PORT", "16379")
    in_compose = inside_compose_network()

    if in_compose:
        influx_url = cfg.get("INFLUX_URL", "http://influxdb:8086")
        redis_addr = cfg.get("REDIS_ADDR", "redis:6379")
    else:
        influx_url = cfg.get("INFLUX_URL", f"http://localhost:{influx_port}")
        if re.search(r"://(influxdb|ftg-influxdb)(:|/|$)", influx_url):
            influx_url = f"http://localhost:{influx_port}"
        redis_addr = cfg.get("REDIS_ADDR", f"localhost:{redis_port}")
        if redis_addr.startswith("redis:"):
            redis_addr = f"localhost:{redis_port}"

    cfg["INFLUX_URL"] = influx_url.rstrip("/")
    cfg["REDIS_ADDR"] = redis_addr

    cfg.setdefault("INFLUX_ORG", "fasttrader")
    cfg.setdefault("INFLUX_TOKEN", "ftg-super-secret-token-change-me")
    cfg.setdefault("INFLUX_BUCKET_RAW", "market_raw")
    cfg.setdefault("FAISS_PATH", "/app/data/faiss_index")
    cfg.setdefault("RETRAIN_REPORT_PATH", "/app/data/retrain_report.json")
    return cfg


def detect_influx_version(base_url: str, token: str) -> int:
    """Return 2 for InfluxDB 2.x, 1 for 1.x."""
    health_url = f"{base_url}/health"
    try:
        req = urllib.request.Request(health_url, method="GET")
        with urllib.request.urlopen(req, timeout=8) as resp:
            body = resp.read().decode("utf-8", errors="replace")
            if resp.status == 200 and "pass" in body.lower():
                return 2
    except urllib.error.HTTPError as exc:
        if exc.code in (401, 403) and token:
            return 2
    except OSError:
        pass

    ping_url = f"{base_url}/ping"
    try:
        req = urllib.request.Request(ping_url, method="GET")
        with urllib.request.urlopen(req, timeout=8) as resp:
            if resp.status == 204:
                return 1
    except OSError:
        pass

    if token:
        return 2
    raise RuntimeError(f"could not detect InfluxDB version at {base_url}")


def parse_rfc3339(ts: str) -> datetime:
    if ts.endswith("Z"):
        ts = ts[:-1] + "+00:00"
    dt = datetime.fromisoformat(ts)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def make_influx_client(cfg: dict[str, str]) -> "InfluxDBClient":
    from influxdb_client import InfluxDBClient

    timeout_ms = int(cfg.get("INFLUX_TIMEOUT_MS", str(DEFAULT_INFLUX_TIMEOUT_SEC * 1000)))
    return InfluxDBClient(
        url=cfg["INFLUX_URL"],
        token=cfg["INFLUX_TOKEN"],
        org=cfg["INFLUX_ORG"],
        timeout=timeout_ms,
    )


def _delete_trade_outcomes_v2(
    client,
    org: str,
    bucket: str,
    start: datetime,
    stop: datetime,
    predicate: str,
    retries: int = 3,
) -> None:
    delete_api = client.delete_api()
    for attempt in range(1, retries + 1):
        try:
            print(f"Influx delete in progress (attempt {attempt}/{retries})...", flush=True)
            delete_api.delete(
                start=start,
                stop=stop,
                bucket=bucket,
                org=org,
                predicate=predicate,
            )
            return
        except Exception as exc:
            retryable = any(
                token in str(exc).lower()
                for token in ("timeout", "timed out", "connection", "reset")
            )
            if attempt >= retries or not retryable:
                raise
            wait = 5 * attempt
            print(
                f"Influx delete failed ({exc}); retrying in {wait}s...",
                file=sys.stderr,
                flush=True,
            )
            time.sleep(wait)


def delete_influx_v2(
    cfg: dict[str, str],
    start: datetime,
    stop: datetime,
    dry_run: bool,
) -> dict[str, Any]:
    url = cfg["INFLUX_URL"]
    token = cfg["INFLUX_TOKEN"]
    org = cfg["INFLUX_ORG"]
    bucket = cfg["INFLUX_BUCKET_RAW"]
    if not token:
        raise RuntimeError("INFLUX_TOKEN is required for InfluxDB 2.x delete")

    predicate = f'_measurement="{TRADE_MEASUREMENT}"'
    summary = {
        "version": 2,
        "bucket": bucket,
        "org": org,
        "start": start.isoformat(),
        "stop": stop.isoformat(),
        "predicate": predicate,
        "timeout_ms": int(cfg.get("INFLUX_TIMEOUT_MS", str(DEFAULT_INFLUX_TIMEOUT_SEC * 1000))),
    }
    if dry_run:
        summary["action"] = "dry_run"
        return summary

    with make_influx_client(cfg) as client:
        print("Counting trade_outcomes before delete...", flush=True)
        before = _count_trade_outcomes_v2(client, org, bucket, start, stop)
        _delete_trade_outcomes_v2(client, org, bucket, start, stop, predicate)
        print("Counting trade_outcomes after delete...", flush=True)
        after = _count_trade_outcomes_v2(client, org, bucket, start, stop)
    summary["deleted_estimate"] = before
    summary["remaining_in_range"] = after
    return summary


def _count_trade_outcomes_v2(client, org: str, bucket: str, start: datetime, stop: datetime) -> int:
    start_s = start.strftime("%Y-%m-%dT%H:%M:%SZ")
    stop_s = stop.strftime("%Y-%m-%dT%H:%M:%SZ")
    flux = f'''
from(bucket: "{bucket}")
  |> range(start: time(v: "{start_s}"), stop: time(v: "{stop_s}"))
  |> filter(fn: (r) => r._measurement == "{TRADE_MEASUREMENT}")
  |> filter(fn: (r) => r._field == "net_pnl")
  |> count()
'''
    total = 0
    tables = client.query_api().query(flux, org=org)
    for table in tables:
        for record in table.records:
            total += int(record.get_value() or 0)
    return total


def delete_influx_v1(cfg: dict[str, str], dry_run: bool) -> dict[str, Any]:
    url = cfg["INFLUX_URL"]
    database = cfg.get("INFLUX_DATABASE") or cfg.get("INFLUX_BUCKET_RAW", "market_raw")
    summary = {
        "version": 1,
        "database": database,
        "measurement": TRADE_MEASUREMENT,
        "query": f'DROP MEASUREMENT "{TRADE_MEASUREMENT}"',
    }
    if dry_run:
        summary["action"] = "dry_run"
        return summary

    query_url = f"{url}/query"
    params = urllib.parse.urlencode({"q": f'DROP MEASUREMENT "{TRADE_MEASUREMENT}"', "db": database})
    user = cfg.get("INFLUX_USER", "")
    password = cfg.get("INFLUX_PASSWORD", "")
    req = urllib.request.Request(f"{query_url}?{params}", method="POST")
    if user:
        import base64

        cred = base64.b64encode(f"{user}:{password}".encode()).decode()
        req.add_header("Authorization", f"Basic {cred}")
    with urllib.request.urlopen(req, timeout=30) as resp:
        body = resp.read().decode("utf-8", errors="replace")
    summary["response"] = body[:500]
    return summary


def flush_redis(cfg: dict[str, str], flushall: bool, dry_run: bool) -> dict[str, Any]:
    import redis

    addr = cfg["REDIS_ADDR"]
    host, _, port = addr.partition(":")
    port_i = int(port or "6379")
    rdb = redis.Redis(host=host, port=port_i, decode_responses=True)
    rdb.ping()

    summary: dict[str, Any] = {"addr": addr, "deleted_keys": []}
    if flushall:
        summary["mode"] = "FLUSHALL"
        if not dry_run:
            rdb.flushall()
        return summary

    summary["mode"] = "selective"
    deleted = 0
    for pattern in REDIS_DELETE_PATTERNS:
        cursor = 0
        while True:
            cursor, keys = rdb.scan(cursor=cursor, match=pattern, count=200)
            if keys:
                summary["deleted_keys"].extend(keys)
                deleted += len(keys)
                if not dry_run:
                    rdb.delete(*keys)
            if cursor == 0:
                break
    summary["deleted_count"] = deleted
    return summary


def reset_local_files(cfg: dict[str, str], dry_run: bool, reset_checkpoints: bool) -> dict[str, Any]:
    """Reset FAISS index + retrain report on disk (container volume or local path)."""
    faiss_base = Path(cfg.get("FAISS_PATH", "/app/data/faiss_index"))
    report = Path(cfg.get("RETRAIN_REPORT_PATH", "/app/data/retrain_report.json"))
    targets = [
        Path(f"{faiss_base}.faiss"),
        Path(f"{faiss_base}.meta.json"),
        report,
    ]
    if reset_checkpoints:
        ckpt_dir = faiss_base.parent / "checkpoints"
        if ckpt_dir.is_dir():
            targets.extend(sorted(ckpt_dir.glob("*.pt")))

    removed: list[str] = []
    for path in targets:
        if path.exists():
            removed.append(str(path))
            if not dry_run:
                path.unlink()

    # If running on host, also try docker exec into ml_engine volume.
    docker_removed: list[str] = []
    if shutil.which("docker") and not Path("/app/data").is_dir():
        docker_paths = [
            "/app/data/faiss_index.faiss",
            "/app/data/faiss_index.meta.json",
            "/app/data/retrain_report.json",
        ]
        if reset_checkpoints:
            docker_paths.append("/app/data/checkpoints")
        if dry_run:
            docker_removed = docker_paths
        else:
            try:
                check = subprocess.run(
                    ["docker", "inspect", "-f", "{{.State.Running}}", ML_CONTAINER],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                if check.stdout.strip() == "true":
                    if reset_checkpoints:
                        subprocess.run(
                            ["docker", "exec", ML_CONTAINER, "rm", "-rf", "/app/data/checkpoints"],
                            check=False,
                            timeout=30,
                        )
                        docker_removed.append("/app/data/checkpoints/")
                    subprocess.run(
                        ["docker", "exec", ML_CONTAINER, "rm", "-f", *docker_paths[:3]],
                        check=False,
                        timeout=30,
                    )
                    docker_removed.extend(docker_paths[:3])
            except (subprocess.SubprocessError, OSError) as exc:
                return {
                    "local_removed": removed,
                    "docker_removed": docker_removed,
                    "docker_warning": str(exc),
                }

    return {"local_removed": removed, "docker_removed": docker_removed}


def main() -> int:
    parser = argparse.ArgumentParser(description="Flush dirty trade-feedback data from FTG stack.")
    parser.add_argument(
        "--env",
        default=str(repo_root() / ".env"),
        help="Path to .env file (default: repo root .env)",
    )
    parser.add_argument(
        "--since",
        default=DEFAULT_SINCE,
        help=f"Influx delete start (RFC3339 UTC). Default: {DEFAULT_SINCE}",
    )
    parser.add_argument(
        "--redis-flushall",
        action="store_true",
        help="FLUSHALL Redis instead of selective key delete (drops screener cache too)",
    )
    parser.add_argument(
        "--reset-checkpoints",
        action="store_true",
        help="Also remove /app/data/checkpoints/*.pt from ml_engine volume",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print actions without mutating data")
    parser.add_argument("--yes", "-y", action="store_true", help="Skip confirmation prompt")
    parser.add_argument(
        "--influx-timeout-sec",
        type=int,
        default=DEFAULT_INFLUX_TIMEOUT_SEC,
        help=f"Influx HTTP timeout in seconds (default: {DEFAULT_INFLUX_TIMEOUT_SEC})",
    )
    args = parser.parse_args()

    cfg = merged_config(Path(args.env))
    cfg["INFLUX_TIMEOUT_MS"] = str(max(args.influx_timeout_sec, 30) * 1000)
    start = parse_rfc3339(args.since)
    stop = datetime.now(timezone.utc)

    print("=== FTG flush_dirty_data ===")
    print(f"env file:     {args.env}")
    print(f"Influx URL:   {cfg['INFLUX_URL']}")
    print(f"Influx org:   {cfg['INFLUX_ORG']}")
    print(f"Influx bucket:{cfg['INFLUX_BUCKET_RAW']}")
    print(f"Redis:        {cfg['REDIS_ADDR']}")
    print(f"Time range:   {start.isoformat()} -> {stop.isoformat()}")
    print(f"Measurement:  {TRADE_MEASUREMENT}")
    print(f"Redis mode:   {'FLUSHALL' if args.redis_flushall else 'selective ' + ', '.join(REDIS_DELETE_PATTERNS)}")
    print(f"Influx timeout:{args.influx_timeout_sec}s")
    print(f"Dry run:      {args.dry_run}")

    if not args.yes and not args.dry_run:
        answer = input("Proceed? [y/N]: ").strip().lower()
        if answer not in ("y", "yes"):
            print("Aborted.")
            return 1

    try:
        version = detect_influx_version(cfg["INFLUX_URL"], cfg.get("INFLUX_TOKEN", ""))
        print(f"InfluxDB version detected: {version}.x")
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    results: dict[str, Any] = {}

    try:
        if version >= 2:
            results["influx"] = delete_influx_v2(cfg, start, stop, args.dry_run)
        else:
            results["influx"] = delete_influx_v1(cfg, args.dry_run)
    except Exception as exc:
        print(f"ERROR: Influx delete failed: {exc}", file=sys.stderr)
        return 3

    try:
        results["redis"] = flush_redis(cfg, args.redis_flushall, args.dry_run)
    except Exception as exc:
        print(f"ERROR: Redis flush failed: {exc}", file=sys.stderr)
        return 4

    results["files"] = reset_local_files(cfg, args.dry_run, args.reset_checkpoints)

    print("\n=== Results ===")
    print(json.dumps(results, indent=2, default=str))
    print("\nDone. Market microstructure data (orderbook_summary/trades) is untouched.")
    print("Restart ml_engine to reload an empty FAISS index:")
    print("  docker compose restart ml_engine")
    return 0


if __name__ == "__main__":
    sys.exit(main())
