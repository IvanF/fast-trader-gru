# Fast Trader GRU — Архитектура системы

_Обновлено: 2026-07-08 14:12 UTC_

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
│  │  Bybit WS    │     │  WS → Redis  │     │                           │    │
│  │  65 symbols  │     │  orderbooks  │     │   ONNX Runtime (GPU)      │    │
│  │              │     │  trades      │     │   FAISS Vector Memory     │    │
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
│     └──────────────┘ │           │ └────────────┘     │                    │
│                       │ gatekeeper│                    │                    │
│                       │ trade_out │                    │                    │
│                       │ market_raw│                    │                    │
│                       └─────┬─────┘                    │                    │
│                             ▲                           │                    │
│                    ┌────────┴─────────┐                 │                    │
│                    │    OMS (Go)      │◀────────────────┘                    │
│                    │                  │                                      │
│                    │  RiskManager     │──────▶ Bybit V5 Demo API           │
│                    │  PositionManager │                                      │
│                    │  ExitGrid        │                                      │
│                    │  DensityExitMgr  │                                      │
│                    │  Time-Stop FSM   │                                      │
│                    │  Smart Labeling  │                                      │
│                    └──────────────────┘                                      │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Pipeline сигналов (ML Engine)

### 2.1 Схема

```
  Market Event (orderbook / trade)
         │
         ▼
  SymbolBuffer.add_orderbook() / add_trade()
         │
         ▼
  ┌──────────────────────────────────────────────────────────┐
  │              _run_tick_prediction()                       │
  │                                                           │
  │  ①  Buffer Warming:      len(points) < 10 → HOLD        │
  │  ②  ONNX Inference:      direction / confidence / vol    │
  │  ③  Toxic Flow Filter:   toxic_prob > 0.40 → HOLD       │
  │  ④  MIN_EDGE:            |pred_pnl| < 0.0025 → HOLD     │
  │  ⑤  Correlation Block:   BTC_corr > 0.70 → HOLD         │
  │  ⑥  Pattern Memory:      cosine_sim ≥ 0.92 + 5 losses   │
  │  ⑦  Symbol Cooldown:     30-60 min after SL              │
  │  ⑧  Dynamic Confidence:  WR < 40% → raise threshold      │
  │  ⑨  Setup Escalation:    3+ losses → threshold 0.80      │
  │  ⑩  Trend Filter:        downtrend → conf = 0.30         │
  │  ⑪  Confidence Threshold: conf ≤ threshold → HOLD        │
  │  ⑫  ExitOptimizer:       SL/TP levels + trade_score      │
  │  ⑬  Trade Score Gate:    score < 0.30 → HOLD             │
  │  ⑭  Event-Driven Gate:   conf ≥ 0.85 OR OBI > 0.4       │
  │  ⑮  Gatekeeper:          P(success) < 0.45 → REJECT     │
  │       │                                                   │
  │       ▼                                                   │
  │  Publish → Redis "orders:signals"                         │
  └──────────────────────────────────────────────────────────┘
```

### 2.2 Формулы

**Direction:**
```
direction = LONG    если pred_pnl > 0
direction = SHORT   если pred_pnl < 0
direction = HOLD    если pred_pnl == 0
```

**Confidence (PnL mode):**
```
confidence = min(|pred_pnl| / 0.01, 1.0)     если |pred_pnl| ≥ 0.0025
confidence = 0                                иначе
```

**MIN_EDGE Filter (UPDATED: 0.0015 → 0.0025):**
```
if |pred_pnl| < 0.0025:
    signal = HOLD (rejected — below noise floor)
```
_Зачем: при commission 0.13% (taker+maker) + spread ~0.05%, edge < 0.25% не покрывает издержки._

**OBI (Order Book Imbalance):**
```
              Σ(top-5 bid_vol) − Σ(top-5 ask_vol)
OBI = ────────────────────────────────────────────────
              Σ(top-5 bid_vol) + Σ(top-5 ask_vol)
```

**Dynamic Confidence Threshold:**
```
penalty = max(1.0, 3.0 − 2.5 × (WR / 0.40))
threshold = min(base_confidence × penalty, 0.95)
```

**Pattern Memory Similarity:**
```
similarity = cosine(state_vector_A, state_vector_B)
block если: similarity ≥ 0.92 И ≥ 5 similar patterns с avg_loss < −$0.10
```

**Gatekeeper (CatBoost):**
```
P(success) = CatBoost.predict_proba(features)
P(success) ≥ 0.45 → PASS
P(success) < 0.45 → REJECT
```

---

## 3. OMS Execution Pipeline

### 3.1 Схема обработки сигнала

```
  Signal (Redis pub/sub)
         │
         ▼
  ┌───────────────────────────────────────────────────────────────┐
  │                     handleSignal()                             │
  │                                                                │
  │  Entry Filters (12 steps):                                     │
  │  Blacklist → DynamicConf → Spread → ZeroDepth → OBI           │
  │  → Momentum → PriceTrend → Volume → FundingRate               │
  │  → Correlation → ExchangeCheck → RiskManager                  │
  │                                                                │
  │  Order Placement:                                              │
  │  conf ≥ 0.95 → Market-Take (immediate)                        │
  │  conf < 0.95 → Limit Chasing (3× 5s)                          │
  └───────────────────────────────────────────────────────────────┘
         │
         ▼
  ActivePosition Created
         │
         ▼
  ┌───────────────────────────────────────────────────────────────┐
  │            Position Lifecycle (500ms tick)                     │
  │                                                                │
  │  ┌───────────────────────────────────────────────────────┐    │
  │  │ Time-Stop FSM (UPDATED):                               │    │
  │  │   Normal mode:  180s (unified, all confidence)         │    │
  │  │   HFT mode:      60s                                   │    │
  │  │                                                        │    │
  │  │   Stage 1: PostOnly Maker (5s passive window)          │    │
  │  │   Stage 2: 5s → Market Kill-Switch                     │    │
  │  │   Zombie:  +60s → force market close                   │    │
  │  └───────────────────────────────────────────────────────┘    │
  │                                                                │
  │  ┌───────────────────────────────────────────────────────┐    │
  │  │ PositionManager (ATR triggers):                        │    │
  │  │   Scale-Out: R ≥ 1.0 → close 50%                      │    │
  │  │   Breakeven: R ≥ 1.5 → SL = fill + fee (90s timer)   │    │
  │  │   Chandelier: R ≥ 2.0 → trailing SL (ATR × 2.5)      │    │
  │  └───────────────────────────────────────────────────────┘    │
  │                                                                │
  │  ┌───────────────────────────────────────────────────────┐    │
  │  │ DensityExitManager:                                    │    │
  │  │   Wall Push: bid/ask > 15x → adjust TP async          │    │
  │  │   Velocity Reversal: momentum > 0.4 → Market exit     │    │
  │  │   Stagnation: hold > 90s + R < 0.15 → breakeven       │    │
  │  └───────────────────────────────────────────────────────┘    │
  │                                                                │
  │  ┌───────────────────────────────────────────────────────┐    │
  │  │ MFE/MAE Tracking + Smart Labeling:                     │    │
  │  │   Every tick: track MaxFavorablePrice / MaxAdversePrice│    │
  │  │   At close: MFE > 75% of TP → label = 1 (override)   │    │
  │  └───────────────────────────────────────────────────────┘    │
  └───────────────────────────────────────────────────────────────┘
```

### 3.2 Risk Management Formulas

**Stop Loss (ATR-based):**
```
SL_distance = max(0.006, min(0.008, ATR_14 × mult))
SL_price = FillPrice ∓ SL_distance
```

**Take Profit:**
```
TP_distance = SL_distance × R:R       # R:R = 1.2
TP_price = FillPrice ± TP_distance
```

**R-Multiple:**
```
R = unrealized_pnl / OriginalRisk
OriginalRisk = |FillPrice − PlannedSL|
```

**Breakeven (90s timer):**
```
SL_breakeven = FillPrice ± 0.0015 × FillPrice
# Only tightens, never widens
```

---

## 4. Time-Stop FSM (UPDATED)

```
  holdSec > hardTimeStopSec
       │
       ▼
  ┌──────────────────────────┐
  │ hardTimeStopSec =         │
  │   HFT:     60s            │
  │   Normal:  180s (all)     │
  └──────────────┬───────────┘
       │
       ▼
  ┌──────────────────────────┐
  │ Profitable at time-stop? │──── YES ──▶ Breakeven Market Close
  └──────────────┬───────────┘
                 │ NO
                 ▼
  ┌──────────────────────────┐
  │ LONG + loss > 0.3%?      │──── YES ──▶ Force Market Close
  └──────────────┬───────────┘
                 │ NO
                 ▼
  ┌──────────────────────────┐
  │ Stage 1: PostOnly Maker  │
  │ Passive limit exit        │
  │ 5-second window           │
  └──────────────┬───────────┘
                 │
       ┌─────────┴──────────┐
       ▼                    ▼
  Filled? YES          Filled? NO (5s)
       │                    │
       ▼                    ▼
    [DONE]          ┌──────────────────┐
                     │ KILL-SWITCH:      │
                     │ Cancel PostOnly   │
                     │ Market Reduce     │
                     │ tryFinalize()     │
                     └──────────────────┘

  Zombie: holdSec > hardTimeStopSec + 60s → emergency market close
```

**Key changes from previous version:**
| Parameter | Before | After |
|-----------|--------|-------|
| Normal time-stop | 240s/150s (high/low conf) | **180s (unified)** |
| Passive window | 30s | **5s** |
| Zombie retry | 120s | **60s** |
| Breakeven timer | 60s | **90s** |
| Kill-Switch prefix | Mixed | `[TIME-STOP FSM]` + `[KILL-SWITCH]` |

---

## 5. ML Architecture

### 5.1 Модельная архитектура

```
  Orderbook (Redis)                    Trade Flow (Redis)
       │                                     │
       ▼                                     ▼
  ┌──────────────────┐           ┌──────────────────────────┐
  │ 2D Tensor 10×10  │           │ Delta Bars (50ms bins)   │
  │ [bid, ask] depth │           │ [Δbid, Δask, buy, sell, │
  └────────┬─────────┘           │  count, velocity]        │
           │                     └────────────┬─────────────┘
           ▼                                  ▼
  ┌──────────────────┐           ┌──────────────────────────┐
  │  orderbook_cnn   │           │  flow_gru_attention       │
  │  Conv2d(2→32,3)  │           │  GRU(6→64) + CrossAttn   │
  │  Conv2d(32→64,3) │           │  Q=flow, K,V=orderbook    │
  │  Linear(64→64)   │           │  → 64d vector             │
  │  → 64d            │           └────────────┬─────────────┘
  └────────┬─────────┘                         │
           └──────────┬───────────────────────┘
                      ▼
           ┌──────────────────┐
           │  Concatenate 128d│
           └────────┬─────────┘
          ┌─────────┴──────────┐
          ▼                    ▼
  ┌──────────────────┐ ┌──────────────────┐
  │  decision_mlp    │ │  exit_optimizer  │
  │  128→64→32→3    │ │  128→32→2       │
  │  → direction     │ │  → sl_pct        │
  │  → confidence    │ │  → tp_pct        │
  │  → vol_mult      │ └──────────────────┘
  └──────────────────┘
           │
           ▼
  ┌──────────────────────────────────────────────┐
  │  AsymmetricPnLLoss (2.5× overestimation)     │
  │                                               │
  │  loss = {  2.5 × (pred − true)²  (overest)  │
  │          {  1.0 × (pred − true)²  (underest) │
  │                                               │
  │  effective_true_pnl = pnl − commissions       │
  │  commissions = 0.13% (taker 0.075% + maker 0.055%) │
  └──────────────────────────────────────────────┘
```

### 5.2 Gatekeeper v2

```
  InfluxDB → gatekeeper_features (20 fields + label)
       │
       ▼
  ┌──────────────────────────────────────┐
  │ gatekeeper_trainer.py (batch)        │
  │                                      │
  │  CatBoost (500 iter, depth=6, AUC)  │
  │  Temporal split 80/20 (no shuffle)   │
  │  Export: gatekeeper_model.cbm        │
  └──────────────────────────────────────┘

  20 features:
  confidence, spread_pct, obi, volume_ratio, momentum,
  price_velocity, atr_pct, funding_rate, btc_correlation,
  volatility_multiplier, symbol_wr, symbol_pnl_sum,
  symbol_consec_losses, hour_of_day, open_positions_count,
  recent_wr_20, pred_pnl

  P(success) ≥ 0.45 → PASS
  P(success) < 0.45 → REJECT
  No model → PASS (fail-safe)

  Current: AUC=0.7842, Acc=79.17%, 423 samples (WR=31.2%)
```

---

## 6. DensityExitManager

```
  ┌─────────────────────────────────────────────────────┐
  │           DensityExitManager (500ms tick)            │
  │                                                      │
  │  Wall Push:                                          │
  │    wall_ratio = bid_depth / ask_depth (LONG)         │
  │    if wall_ratio > 15 → adjust TP to current ± 0.2% │
  │    → goroutine: cancel old TP, place new limit       │
  │                                                      │
  │  Velocity Reversal:                                  │
  │    momentum = OBI_velocity                           │
  │    shift = PressureShift(0.3)                        │
  │    LONG + shift=-1 + mom < −0.4 → Market Kill-Switch │
  │    SHORT + shift=+1 + mom > 0.4 → Market Kill-Switch │
  │                                                      │
  │  Stagnation:                                         │
  │    holdSec > 90 AND currentR < 0.15                  │
  │    → SL = fillPrice ± 0.15% fee buffer               │
  └─────────────────────────────────────────────────────┘
```

---

## 7. Smart Labeling (MFE)

```
  During position lifetime (every 500ms tick):
  LONG:  MaxFavorablePrice = max(mid, prev)
  SHORT: MaxFavorablePrice = min(mid, prev)

  At close (finalizeClose):
  ┌──────────────────────────────────────────────┐
  │ if pnl ≤ 0 AND MaxFavorablePrice > 0:        │
  │   tpDist  = |TP − FillPrice|                  │
  │   mfeDist = |MaxFavorable − FillPrice|        │
  │   mfePct  = mfeDist / tpDist                  │
  │                                               │
  │   if mfePct > 0.75:                           │
  │     label = 1  (entry accurate, bad exec)     │
  │     log: [LABEL CLEANUP]                      │
  │   else:                                       │
  │     label = 0                                 │
  │ else:                                         │
  │   label = (pnl > 0) ? 1 : 0                  │
  └──────────────────────────────────────────────┘
```

---

## 8. Data Flow

```
Bybit WS ──▶ Screener ──▶ Ingestion ──▶ Redis ──▶ ML Engine
                                                │
                                                ▼
                                        ONNX Inference
                                        (orderbook_cnn + flow_gru + decision_mlp)
                                                │
                                                ▼
                                        Signal Pipeline (15 gates)
                                        + Gatekeeper (CatBoost)
                                                │
                                     Redis "orders:signals"
                                                │
                                                ▼
                                         OMS Execution
                                        (12 entry filters)
                                                │
                                    Bybit V5 Demo API
                                                │
                                     Redis "execution:results"
                                     ┌──────────┴──────────┐
                                     ▼                     ▼
                              ML Engine               OMS finalizeClose
                              (online learning)      (Smart Labeling)
                                     │                     │
                                     ▼                     ▼
                              Pattern Memory         InfluxDB
                              FAISS vectors          ├─ gatekeeper_features
                                                     ├─ trade_outcomes
                                                     └─ market_raw
```

---

## 9. Entry Filters

| # | Filter | Threshold | Action |
|---|--------|-----------|--------|
| 1 | Blacklist | ETF tokens | REJECT |
| 2 | Dynamic Confidence | WR < 40% → raise | REJECT |
| 3 | Spread | > 0.5% | REJECT |
| 4 | Zero Depth | bid+ask = 0 | REJECT |
| 5 | OBI Direction | ±0.4 | REJECT |
| 6 | Momentum (shift) | against direction | REJECT |
| 7 | Price Trend | 30s move > 0.5% | REJECT |
| 8 | Volume | < 0.5× SMA(20) | REJECT |
| 9 | Funding Rate | ±0.05% | REJECT |
| 10 | Correlation | > 0.70 + conf < 0.95 | HOLD |
| 11 | Exchange Cross-Check | no position | REJECT |
| 12 | RiskManager | SL constraints | REJECT |

---

## 10. Key Constants

| Constant | Value | Description |
|----------|-------|-------------|
| MIN_EDGE | **0.0025** | Min expected PnL (0.25%) — **UPDATED** |
| GK_THRESHOLD | 0.45 | Gatekeeper threshold |
| PATTERN_SIMILARITY | 0.92 | Cosine similarity threshold |
| R:R | 1.2 | Risk:Reward ratio |
| MIN_SL_PCT | 0.006 | Min SL (0.6%) |
| MAX_SL_PCT | 0.008 | Max SL (0.8%) |
| MAX_CORRELATION | 0.70 | BTC correlation limit |
| VOL_MULTIPLIER_CAP | 2.0 | Max vol_mult |
| NORMAL_TSTOP | **180s** | Time-stop (all confidence) — **UPDATED** |
| HFT_TSTOP | 60s | Time-stop (HFT mode) |
| PASSIVE_WINDOW | **5s** | PostOnly → Market timeout — **UPDATED** |
| ZOMBIE_RETRY | **60s** | Zombie force close — **UPDATED** |
| BREAKEVEN_SEC | **90s** | Breakeven timer — **UPDATED** |
| MFE_THRESHOLD | 0.75 | Smart labeling override |
| DENSITY_WALL | 15.0 | Wall push threshold |
| DENSITY_STALL | 90s | Stagnation time |
| DENSITY_STALL_R | 0.15 | Stagnation R threshold |

---

## 11. Infrastructure

| Component | Tech | Container |
|-----------|------|-----------|
| Screener | Go | ftg-screener |
| Ingestion | Go | ftg-ingestion |
| ML Engine | Python + CUDA | ftg-ml-engine |
| OMS | Go | ftg-oms |
| History Logger | Go | ftg-history-logger |
| Redis | 6.x | ftg-redis |
| InfluxDB | 2.x | ftg-influxdb |
| Prometheus | Go | ftg-prometheus |
| Grafana | Go | ftg-grafana |

**Trading:** $2,000 balance, 5× leverage, $10 margin, 200 max concurrent, R:R=1.2
