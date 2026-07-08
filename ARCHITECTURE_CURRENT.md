# Fast Trader GRU — Архитектура системы

_Обновлено: 2026-07-08 16:05 UTC_

---

## 1. Общая архитектура

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         FAST TRADER GRU                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────────────────┐    │
│  │   Screener   │────▶│  Ingestion   │────▶│       ML Engine           │    │
│  │     (Go)     │     │     (Go)     │     │   (Python 3.10 + CUDA)   │    │
│  │  Bybit WS    │     │  WS → Redis  │     │   ONNX Runtime (GPU)      │    │
│  │  68 symbols  │     │  orderbooks  │     │   FAISS (4971 vectors)    │    │
│  └──────────────┘     └──────┬───────┘     │   CatBoost Gatekeeper     │    │
│                              │              │   DensityExitManager      │    │
│                              ▼              └──────────┬───────────────┘    │
│                    ┌──────────────────┐                 │                    │
│                    │    Redis 6.x     │◀────────────────│                    │
│                    │  Central Bus     │                 │                    │
│                    │  market:orderbook│   orders:signals│                    │
│                    │  market:trades   │────────────────▶│                    │
│                    │  execution:results│◀───────────────│                    │
│                    └────────┬─────────┘                 │                    │
│              ┌──────────────┼──────────────┐            │                    │
│              ▼              ▼              ▼            │                    │
│     ┌──────────────┐ ┌───────────┐ ┌────────────┐     │                    │
│     │  History     │ │  InfluxDB │ │ Prometheus │     │                    │
│     │  Logger      │ │  TSDB     │ │  Metrics   │     │                    │
│     └──────────────┘ │ gatekeeper│ └────────────┘     │                    │
│                       │ trade_out │                    │                    │
│                       │ market_raw│                    │                    │
│                       └─────┬─────┘                    │                    │
│                             ▲                           │                    │
│                    ┌────────┴─────────┐                 │                    │
│                    │    OMS (Go)      │◀────────────────┘                    │
│                    │                  │                                      │
│                    │  StateMachine    │                                      │
│                    │  RiskManager     │──────▶ Bybit V5 Demo API           │
│                    │  PositionManager │                                      │
│                    │  ExitGrid        │                                      │
│                    │  DensityExitMgr  │                                      │
│                    │  Time-Stop FSM   │                                      │
│                    │  Smart Labeling  │                                      │
│                    │  Stale Guard     │                                      │
│                    └──────────────────┘                                      │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Pipeline сигналов (ML Engine → OMS)

```
  Market Event
       │
       ▼
  ┌──────────────────────────────────────────────────────────────┐
  │              _run_tick_prediction()                           │
  │                                                               │
  │  ①  Buffer Warming:       len(points) < 10 → HOLD           │
  │  ②  ONNX Inference:       direction / conf / vol             │
  │  ③  [KNIFE-GUARD]:        BTC vel < -0.0005 + corr > 0.6    │
  │                            AND direction=LONG → HOLD          │
  │  ④  [KNIFE-GUARD]:        OBI < -0.1 + direction=LONG → HOLD│
  │  ⑤  Toxic Flow Filter:    toxic > 0.40 → HOLD               │
  │  ⑥  MIN_EDGE:             |pred_pnl| < 0.0025 → HOLD       │
  │  ⑦  Correlation Block:    BTC_corr > 0.70 → HOLD            │
  │  ⑧  Pattern Memory:       cosine ≥ 0.92 + 5 losses → BLOCK │
  │  ⑨  Symbol Cooldown:      30-60 min after SL                 │
  │  ⑩  Dynamic Confidence:   WR < 40% → raise threshold         │
  │  ⑪  Setup Escalation:     3+ losses → threshold 0.80         │
  │  ⑫  Trend Filter:         downtrend → conf = 0.30            │
  │  ⑬  Confidence Threshold: conf ≤ threshold → HOLD            │
  │  ⑭  ExitOptimizer:        SL/TP levels + trade_score         │
  │  ⑮  Trade Score Gate:     score < 0.30 → HOLD               │
  │  ⑯  Event-Driven Gate:    conf ≥ 0.85 OR OBI > 0.4          │
  │  ⑰  Gatekeeper (asymmetric):                                 │
  │       LONG:  P(success) ≥ 0.65 → PASS                       │
  │       SHORT: P(success) ≥ 0.45 → PASS                       │
  │  Publish → Redis "orders:signals"                             │
  └──────────────────────────────────────────────────────────────┘
```

### Ключевые формулы

**Confidence:**
```
confidence = min(|pred_pnl| / 0.01, 1.0)     если |pred_pnl| ≥ 0.0025
confidence = 0                                иначе
```

**OBI (Order Book Imbalance):**
```
              Σ(top-5 bid_vol) − Σ(top-5 ask_vol)
OBI = ────────────────────────────────────────────────
              Σ(top-5 bid_vol) + Σ(top-5 ask_vol)
```

**Dynamic Confidence Threshold:**
```
penalty = max(1.0, 3.0 − 2.5 × (WR / 0.40))
threshold = min(base × penalty, 0.95)
```

**BTC Knife Guard:**
```
block LONG если: corr > 0.60 И btc_1m_trend < -0.0005
block LONG если: OBI < -0.1 (asks доминируют)
```

---

## 3. OMS Execution Pipeline

```
  Signal (Redis)
       │
       ▼
  ┌───────────────────────────────────────────────────────────────┐
  │  Entry Filters (12 steps):                                     │
  │  Blacklist → DynamicConf → Spread → ZeroDepth → OBI           │
  │  → Momentum → PriceTrend → Volume (asymmetric!) → FundingRate │
  │  → Correlation → LONG Cluster (max 2) → ExchangeCheck          │
  │  → RiskManager                                                │
  └───────────────────────────────────────────────────────────────┘
       │
       ▼
  ActivePosition Created (State = StateActive)
       │
       ▼
  ┌───────────────────────────────────────────────────────────────┐
  │  Position Lifecycle (500ms tick)                                │
  │                                                                │
  │  ┌───────────────────────────────────────────────────────┐    │
  │  │ [STALE GUARD] — holdMs < 60s → skip ghost check       │    │
  │  │  Prevents premature removal from API lag               │    │
  │  └───────────────────────────────────────────────────────┘    │
  │                                                                │
  │  ┌───────────────────────────────────────────────────────┐    │
  │  │ Time-Stop FSM (State Machine):                         │    │
  │  │  StateActive → StateClosingPassive → StateClosingAggr  │    │
  │  │                                                       │    │
  │  │  Normal: 180s, HFT: 60s                               │    │
  │  │  Stage 1: PostOnly Maker (5s window)                  │    │
  │  │  Stage 2: Market Kill-Switch (atomic transition)       │    │
  │  │  Zombie: +60s → emergency close                        │    │
  │  └───────────────────────────────────────────────────────┘    │
  │                                                                │
  │  ┌───────────────────────────────────────────────────────┐    │
  │  │ PositionManager (ATR):                                 │    │
  │  │  Scale-Out R≥1.0, Breakeven R≥1.5 (90s), Chandelier   │    │
  │  └───────────────────────────────────────────────────────┘    │
  │                                                                │
  │  ┌───────────────────────────────────────────────────────┐    │
  │  │ DensityExitManager:                                    │    │
  │  │  Wall Push ratio>15 → adjust TP async                  │    │
  │  │  Velocity Reversal → Market Kill-Switch                │    │
  │  │  Stagnation hold>90s + R<0.15 → breakeven              │    │
  │  └───────────────────────────────────────────────────────┘    │
  │                                                                │
  │  ┌───────────────────────────────────────────────────────┐    │
  │  │ MFE/MAE + Smart Labeling:                              │    │
  │  │  Track MaxFavorablePrice every tick                    │    │
  │  │  At close: MFE > 75% TP → label=1 (override)          │    │
  │  └───────────────────────────────────────────────────────┘    │
  └───────────────────────────────────────────────────────────────┘
```

### Asymmetric Volume Filter
```
LONG:  current_vol ≥ 1.2 × SMA(20)   (active buying interest)
SHORT: current_vol ≥ 0.5 × SMA(20)   (dead market)
```

### LONG Cluster Guard
```
block LONG если: ≥ 2 коррелированных LONG позиции с BTC_corr > 0.70
```

---

## 4. State Machine (Atomic)

```go
type ActivePositionState int32

const (
    StateActive             = 0
    StateClosingPassive     = 1
    StateClosingAggressive  = 2
    StateClosed             = 3
)
```

**Thread-Safe Transitions:**
```go
atomic.CompareAndSwapInt32(&pos.State, StateActive, StateClosingPassive)
atomic.StoreInt32(&pos.State, StateClosingAggressive)
atomic.StoreInt32(&pos.State, StateClosed)
```

**State Snapshot Pattern (Lock-Decoupled):**
```
1. s.mu.Lock() → copy fields → s.mu.Unlock()
2. Network I/O (no lock held)
3. atomic.StoreInt32(&pos.State, newState)
```

---

## 5. Time-Stop FSM

```
  holdSec > hardTimeStopSec
       │
       ▼
  ┌──────────────────────┐
  │  Normal: 180s        │
  │  HFT: 60s            │
  └──────────┬───────────┘
             ▼
  ┌──────────────────────┐
  │  Profitable?         │──── YES ──▶ Market close (breakeven)
  └──────────┬───────────┘
             │ NO
             ▼
  ┌──────────────────────┐
  │  LONG + loss > 0.3%? │──── YES ──▶ Force market close
  └──────────┬───────────┘
             │ NO
             ▼
  ┌──────────────────────┐
  │  CAS: Active →        │
  │  ClosingPassive       │
  └──────────┬───────────┘
             ▼
  ┌──────────────────────┐
  │  Stage 1: PostOnly   │
  │  5s passive window    │
  └──────────┬───────────┘
             │
    ┌────────┴────────┐
    ▼                 ▼
  Filled?          5s elapsed
  YES → DONE      ▼
              ┌──────────────────┐
              │ CAS: →           │
              │ ClosingAggressive│
              │ Cancel + Market  │
              │ Kill-Switch      │
              └──────────────────┘

  Zombie: holdSec > hardTime + 60s → CAS → market close
```

---

## 6. DensityExitManager

```
  Wall Push:    bid/ask ratio > 15 → adjust TP (±0.2%)
  Velocity:     momentum > 0.4 + shift reversal → Market exit
  Stagnation:   hold > 90s + R < 0.15 → breakeven SL
```

---

## 7. Smart Labeling (MFE)

```
  At close:
  if pnl ≤ 0 AND MaxFavorablePrice > 0:
    mfePct = |MaxFavorable - Fill| / |TP - Fill|
    if mfePct > 0.75: label = 1 (entry accurate, bad execution)
  else: label = (pnl > 0) ? 1 : 0
```

---

## 8. Stale Guard

```
  evaluatePosition():
    holdMs < 60s → return (skip ghost check)
    holdMs ≥ 60s → syncPositionFromExchange()
      hasPos=false → handleGhostPosition()
        ensureExchangeFlat() (6 retries)
        GetRecentClosedPnL() → actual PnL
        finalizeClose(actual_pnl, reason="stale_tracker_removed")
```

---

## 9. Gatekeeper v2 (Asymmetric)

```
  20 features → CatBoost (AUC=0.7842, Acc=79.17%)
  576 samples (191 wins / 385 losses, WR=33.2%)

  LONG signals:  threshold = 0.65 (conservative)
  SHORT signals: threshold = 0.45 (permissive)
  No model: pass-through (fail-safe)
```

---

## 10. Key Constants

| Constant | Value | Description |
|----------|-------|-------------|
| MIN_EDGE | **0.0025** | Min expected PnL (0.25%) |
| NORMAL_TSTOP | **180s** | Unified time-stop |
| HFT_TSTOP | 60s | HFT time-stop |
| PASSIVE_WINDOW | **5s** | PostOnly → Kill-Switch |
| ZOMBIE_RETRY | **60s** | Zombie force close |
| BREAKEVEN_SEC | **90s** | Breakeven timer |
| STALE_GUARD | **60s** | Grace period before ghost check |
| GK_LONG | **0.65** | LONG Gatekeeper threshold |
| GK_SHORT | **0.45** | SHORT Gatekeeper threshold |
| LONG_VOL_SMA | **1.2** | LONG volume requirement |
| SHORT_VOL_SMA | 0.5 | SHORT volume requirement |
| LONG_CLUSTER | **2** | Max correlated LONG positions |
| BTC_VEL_THRESH | -0.0005 | BTC trend filter |
| OBI_NEG_THRESH | -0.1 | OBI LONG block |
| MFE_THRESHOLD | 0.75 | Smart label override |
| DENSITY_WALL | 15.0 | Wall push ratio |
| DENSITY_STALL | 90s | Stagnation time |
| DENSITY_STALL_R | 0.15 | Stagnation R |
| R:R | 1.2 | Risk:Reward |
