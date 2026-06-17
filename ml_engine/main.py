"""ML Engine entrypoint."""

from __future__ import annotations

import os

# Must be set before onnxruntime is imported (Docker cpuset + HT causes affinity errors)
os.environ.setdefault("ORT_DISABLE_CPU_AFFINITY", "1")
os.environ.setdefault("OMP_NUM_THREADS", os.getenv("ORT_INTRA_OP_THREADS", "4"))
os.environ.setdefault("MKL_NUM_THREADS", os.getenv("ORT_INTRA_OP_THREADS", "4"))

import asyncio
import logging
import sys

# Ensure src is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import Config
from src.engine import MLEngine

logging.basicConfig(
    level=logging.INFO,
    format='{"ts":"%(asctime)s","level":"%(levelname)s","msg":"%(message)s"}',
)


async def main() -> None:
    cfg = Config.load()
    engine = MLEngine(cfg)
    await engine.run()


if __name__ == "__main__":
    try:
        import uvloop
        uvloop.install()
    except ImportError:
        pass
    asyncio.run(main())
