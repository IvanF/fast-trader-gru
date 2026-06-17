# Fast Trader GRU

Microservices algorithmic trading infrastructure for **Bybit USDT Perpetual Futures (V5 API)**. Redis Pub/Sub is the central data bus. Go services handle screening, ingestion, and execution; Python handles ML inference with ONNX Runtime (CUDA) and FAISS vector memory.

## Architecture

```
┌─────────────┐     config:active_symbols      ┌──────────────────┐
│  Screener   │ ─────────────────────────────► │  Data Ingestion  │
│    (Go)     │                                │  WS → Redis Pub  │
└─────────────┘                                └────────┬─────────┘
                                                        │
                        market:orderbook:* / market:trades:*
                                                        ▼
                                               ┌──────────────────┐
                                               │    ML Engine     │
                                               │ Python + ONNX GPU│
                                               │ FAISS + Regime   │
                                               └────────┬─────────┘
                                                        │
                                                   orders:signals
                                                        ▼
                                               ┌──────────────────┐
                                               │  OMS Execution   │
                                               │ Grid + Risk (Go) │
                                               └────────┬─────────┘
                                                        │
                                                  execution:results
                                                        ▼
                                               ┌──────────────────┐
                                               │   ML Engine      │
                                               │ Continuous Learn │
                                               └──────────────────┘
```

## Services

| Service | Language | Role |
|---------|----------|------|
| `screener` | Go | 60-min loop: filter USDT perps by turnover ≥ 50M, funding rate bounds |
| `data_ingestion` | Go | Dynamic WS manager for `orderbook.50` + `publicTrade` streams |
| `ml_engine` | Python 3.11 | Feature engineering, ONNX late-fusion, FAISS memory, regime detection |
| `oms_execution` | Go | Adaptive grid maker, liquidity wall SL, partial TP, time-stop guard |
| `history_logger` | Go | Redis → InfluxDB batch writer (ticks + L2 depth) |
| `influxdb` | — | Historical data lake (14d raw, 365d aggregated features) |
| `redis` | — | In-memory pub/sub bus (AOF disabled) |
| `prometheus` + `grafana` | — | Full observability stack |

## Hardware Pinning (cpuset)

| Container | CPU Cores |
|-----------|-----------|
| Redis | 0 |
| Data Ingestion | 1 |
| OMS Execution | 2 |
| ML Engine | 3–6 |
| History Logger | 7 |

## Historical Data Lake (InfluxDB 2.x)

| Bucket | Retention | Contents |
|--------|-----------|----------|
| `market_raw` | 14 days (336h) | High-frequency trades + L2 orderbook depth |
| `market_features` | 365 days | 1-minute downsampled OBI, volumes, trade metrics |

`history_logger` subscribes to `market:orderbook:*` and `market:trades:*`, encodes Line Protocol, and flushes asynchronously every **5 seconds** or **10,000 points** (non-blocking enqueue).

### Offline Training (PnL Join + Hot-Swap)

Each closed trade from `execution:results` is written to InfluxDB as `trade_outcomes` and joined with market features (±300s window) for real labels.

```bash
# Export joined dataset
docker compose exec ml_engine python scripts/query_influx_training_data.py \
  --symbol BTCUSDT --start -30d --output /app/data/training/btc_joined.npz

# Full train + atomic ONNX promote + hot-swap (no service restart)
docker compose exec ml_engine python scripts/train.py \
  --symbols BTCUSDT,ETHUSDT --days 30 --epochs 15

# Daily cron pipeline
docker compose exec ml_engine python scripts/daily_retrain.py \
  --symbols BTCUSDT,ETHUSDT --days 30
```

Hot-swap triggers:
- Redis channel `models:reload` (published by `train.py`)
- Manifest + ONNX file watch every 10s (`manifest.json` mtime)

### Rolling Retrain Feedback Loop

Built into `ml_engine` — no separate cron container required.

| Trigger | Default |
|---------|---------|
| Time interval | Every **2 hours** |
| Trade count | Every **100** closed trades on `execution:results` |

Training runs as `nice -n 15` subprocess so live inference keeps CPU/GPU priority.

```bash
# Manual rolling retrain (6h Influx lookback, asymmetric PnL loss)
docker compose exec ml_engine python scripts/train.py \
  --symbols BTCUSDT,ETHUSDT --hours 6 --incremental
```

Prometheus: `ml_retrain_duration_seconds`, `ml_loss_delta`, `onnx_hot_swap_success_total`, `last_training_timestamp`.

## Quick Start

### Prerequisites

- Docker & Docker Compose v2
- NVIDIA Container Toolkit (for ML Engine GPU)
- Bybit API key with trade permissions

### Launch

```bash
cp .env.example .env
# Edit .env with your BYBIT_API_KEY and BYBIT_API_SECRET

docker compose up --build -d
```

### Endpoints

Host ports use the **FTG block** (`FTG_*` in `.env`) to avoid collisions with other stacks (e.g. `infra-redis` on `:6379`, `sen-*` on `:80xx`).

| Service | Host URL | Container port |
|---------|----------|----------------|
| Grafana | http://localhost:13000 (admin / admin) | 3000 |
| Prometheus | http://localhost:19090 | 9090 |
| InfluxDB UI | http://localhost:18086 | 8086 |
| Redis (debug) | localhost:16379 | 6379 |
| Screener metrics | internal only `:9100/metrics` | 9100 |
| Ingestion metrics | internal only `:9101/metrics` | 9101 |
| OMS metrics | internal only `:9102/metrics` | 9102 |
| ML metrics | internal only `:9103/metrics` | 9103 |
| History Logger metrics | internal only `:9104/metrics` | 9104 |

Override in `.env`: `FTG_REDIS_PORT`, `FTG_INFLUX_PORT`, `FTG_PROMETHEUS_PORT`, `FTG_GRAFANA_PORT`.

## Redis Channels

| Channel | Publisher | Consumer |
|---------|-----------|----------|
| `config:active_symbols` | Screener | Ingestion, ML Engine |
| `market:orderbook:[SYMBOL]` | Ingestion | ML Engine, OMS |
| `market:trades:[SYMBOL]` | Ingestion | ML Engine |
| `orders:signals` | ML Engine | OMS |
| `execution:results` | OMS | ML Engine |

## ML Pipeline

1. **Feature Engineering** — 300s rolling buffer per symbol: OBI, CVD, order flow speed, VWAP deviation, 15m/1h macro trends, funding rate.
2. **Regime Detection** — EMA volatility + GMM every 60s → Trending / Choppy / Breakout.
3. **ONNX Inference** — 1D-CNN (orderbook) + GRU + Self-Attention (flow) → Master State Vector \(V_{state}\).
4. **FAISS Memory** — Top-10 FlatIP matches with time-decay and regime-aware forgetting → \(V_{memory}\).
5. **Decision MLP** — Fuses \(V_{state}\) + \(V_{memory}\) → LONG/SHORT/HOLD, confidence, volatility multiplier.
6. **Correlation Filter** — 5-min Pearson vs BTC; if > 0.85 → HOLD or 25% position scale.

## OMS Risk Features

- **Liquidity Wall SL** — `find_peaks` on orderbook depth; SL placed 2 ticks behind largest wall.
- **Asymmetric Profit Taking** — At 1R, close 45%, move SL to breakeven, activate CVD-based trailing stop.
- **Time-Stop Guard** — Force-close after configurable N seconds if TP not hit.
- **Maker-only entry** — PostOnly limit orders via Bybit V5 REST.

## Re-export ONNX Models

```bash
cd ml_engine
pip install torch
python scripts/export_models.py
```

## Development (local)

```bash
# Go services
cd screener && go run ./cmd/screener
cd data_ingestion && go run ./cmd/ingestion
cd oms_execution && go run ./cmd/oms

# Python ML engine
cd ml_engine && python main.py
```

## License

Proprietary — internal use only.
