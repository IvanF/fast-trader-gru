"""Backfill FAISS knowledge base from InfluxDB trade outcomes.

Reads historical trade outcomes with state vectors and populates FAISS index.
This dramatically accelerates learning by seeding the memory with past experience.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

STATE_DIM = int(os.getenv("STATE_DIM", "128"))
FAISS_PATH = os.getenv("FAISS_PATH", "/app/data/faiss_index")
INFLUX_URL = os.getenv("INFLUX_URL", "http://influxdb:8086")
INFLUX_TOKEN = os.getenv("INFLUX_TOKEN", "ftg-super-secret-token-change-me")
INFLUX_ORG = os.getenv("INFLUX_ORG", "fasttrader")
INFLUX_BUCKET = os.getenv("INFLUX_BUCKET_RAW", "market_raw")
LOOKBACK_HOURS = int(os.getenv("BACKFILL_LOOKBACK_HOURS", "168"))  # 7 days
BATCH_SIZE = 500


def query_trade_outcomes():
    """Query InfluxDB for trade outcomes with state vectors."""
    from influxdb_client import InfluxDBClient
    from influxdb_client.client.write_api import SYNCHRONOUS

    client = InfluxDBClient(url=INFLUX_URL, token=INFLUX_TOKEN, org=INFLUX_ORG)
    query_api = client.query_api()

    query = f'''
    from(bucket: "{INFLUX_BUCKET}")
      |> range(start: -{LOOKBACK_HOURS}h)
      |> filter(fn: (r) => r._measurement == "trade_outcomes")
      |> filter(fn: (r) => r._field == "net_pnl" or r._field == "state_vector")
      |> pivot(rowKey: ["_time", "symbol", "direction", "regime", "close_reason"],
               columnKey: ["_field"],
               valueColumn: "_value")
    '''

    logger.info("querying InfluxDB for trade outcomes (lookback=%dh)...", LOOKBACK_HOURS)
    tables = query_api.query(query, org=INFLUX_ORG)

    results = []
    for table in tables:
        for record in table.records:
            try:
                pnl = record.values.get("net_pnl")
                state_vec = record.values.get("state_vector")
                direction = record.values.get("direction", "HOLD")
                regime = record.values.get("regime", "Choppy")
                symbol = record.values.get("symbol", "")
                ts = record.get_time().timestamp()

                if pnl is None:
                    continue

                pnl = float(pnl)

                if state_vec is not None:
                    if isinstance(state_vec, str):
                        vec = np.array(json.loads(state_vec), dtype=np.float32)
                    elif isinstance(state_vec, list):
                        vec = np.array(state_vec, dtype=np.float32)
                    else:
                        continue
                else:
                    vec = np.random.randn(STATE_DIM).astype(np.float32)
                    vec = vec / (np.linalg.norm(vec) + 1e-8)

                if vec.size < STATE_DIM:
                    vec = np.pad(vec, (0, STATE_DIM - vec.size))
                vec = vec[:STATE_DIM]

                results.append({
                    "vector": vec,
                    "pnl": pnl,
                    "regime": regime or "Choppy",
                    "direction": direction or "HOLD",
                    "symbol": symbol,
                    "timestamp": ts,
                })
            except Exception as exc:
                logger.debug("skip record: %s", exc)
                continue

    client.close()
    return results


def backfill():
    """Main backfill logic."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from memory import ExperienceEngine

    outcomes = query_trade_outcomes()
    logger.info("found %d trade outcomes in InfluxDB", len(outcomes))

    if not outcomes:
        logger.warning("no outcomes to backfill")
        return

    memory = ExperienceEngine(STATE_DIM, FAISS_PATH)
    initial_size = memory.size
    logger.info("current FAISS size: %d", initial_size)

    added = 0
    skipped = 0
    for i, outcome in enumerate(outcomes):
        vid = memory.add(
            outcome["vector"],
            outcome["pnl"],
            outcome["regime"],
            outcome["direction"],
        )
        if vid >= 0:
            added += 1
        else:
            skipped += 1

        if (i + 1) % 100 == 0:
            logger.info("progress: %d/%d (added=%d skipped=%d)", i + 1, len(outcomes), added, skipped)

    memory.persist()
    final_size = memory.size

    logger.info(
        "backfill complete: added=%d skipped=%d total=%d (was %d, +%d)",
        added, skipped, final_size, initial_size, final_size - initial_size,
    )

    stats = {
        "timestamp": time.time(),
        "outcomes_queried": len(outcomes),
        "added": added,
        "skipped": skipped,
        "initial_size": initial_size,
        "final_size": final_size,
    }
    report_path = os.path.join(os.path.dirname(FAISS_PATH), "backfill_report.json")
    with open(report_path, "w") as f:
        json.dump(stats, f, indent=2)
    logger.info("report saved to %s", report_path)


if __name__ == "__main__":
    backfill()
