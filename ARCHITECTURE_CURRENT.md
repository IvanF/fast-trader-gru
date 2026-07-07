# Fast Trader GRU — Текущая архитектура (2026-07-06)

---

## 1. Общая схема системы

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                          Bybit V5 WebSocket (данные)                            │
│  Orderbook L2 (20 уровней bid/ask/size)                                         │
│  Trade Flow (price/size/direction)                                              │
│  Funding Rate + Price History                                                   │
└────────────────────────────────────┬────────────────────────────────────────────┘
                                     │ Redis Pub/Sub (market:orderbook:*)
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                        ML ENGINE (Python 3.10 + ONNX GPU)                       │
│                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │ FeatureStore (50ms Delta Bars + 22 Macro Features)                     │   │
│  │                                                                         │   │
│  │ DeltaBarEncoder: 6→64→64  │  Macro: 22→64 (macro_proj)                │   │
│  │ (delta_bid/ask_vol,       │                                             │   │
│  │  buy/sell_vol, count,     │                                             │   │
│  │  price_velocity)          │                                             │   │
│  │                           │                                             │   │
│  │ OrderbookCNN              │  FlowGRU + Attention                       │   │
│  │ Conv1d (2,60→32)          │  GRU (3,60→32) + SelfAttention            │   │
│  │ Conv2d (1×20×60→32)       │                                             │   │
│  └───────────┬───────────────┴────────────────────────────────────────────┘   │
│              │                                                                 │
│              ▼                                                                 │
│  ┌──────────────────────────────────────────────────────────────────────────┐  │
│  │ FusionModel (backward-compat): CNN(32)+GRU(32)+Macro(22→64) → 128-dim  │  │
│  └───────────────────────────┬──────────────────────────────────────────────┘  │
│                              │                                                  │
│                              ▼                                                  │
│  ┌──────────────────────────────────────────────────────────────────────────┐  │
│  │ DecisionMLP (shared backbone → 3 heads):                                │  │
│  │   Head 1: pred_pnl (float) — Expected PnL                               │  │
│  │   Head 2: trap_logit → trap_prob (0-1)                                  │  │
│  │   Head 3: toxic_logit → toxic_flow_prob (0-1)                           │  │
│  │                                                                          │  │
│  │ direction = LONG if pred_pnl > 0 else SHORT                             │  │
│  │ confidence = min(|pred_pnl| / 0.01, 1.0)                               │  │
│  └──────────────────────────────────────────────────────────────────────────┘  │
│                                                                                 │
│  ┌──────────────────────┐  ┌───────────────────────────────────────────────┐   │
│  │ FAISS Memory         │  │ Pattern Memory                                │   │
│  │ 2,462 winning trades │  │ cosine_sim ≥ 0.92 → block                    │   │
│  │ → 8-dim v_memory     │  │ TTL: 24h, min 3 similar losses              │   │
│  └──────────────────────┘  └───────────────────────────────────────────────┘   │
│                                                                                 │
│  ┌──────────────────────────────────────────────────────────────────────────┐  │
│  │ Online Learner (EWC) + Replay Buffer (200) + Retrain Worker             │  │
│  │ Retrain: every 10 winning trades OR every 2h, 48h lookback, 12 epochs  │  │
│  │ Loss: AsymmetricPnLLoss (2.5× overestimation penalty, fee-adjusted)    │  │
│  └──────────────────────────────────────────────────────────────────────────┘  │
│                                                                                 │
│  ФИЛЬТРЫ ДО ОТПРАВКИ В OMS:                                                    │
│                                                                                 │
│  1. Toxic flow:      toxic_prob > 0.40 → HOLD                                 │
│  2. MIN_EDGE:        |pred_pnl| < 0.0025 → HOLD                               │
│  3. Pattern memory:  3+ similar losses (cosine ≥ 0.92) → block                │
│  4. Symbol+Setup:    3+ losses avg<-$0.10 → threshold 0.60                   │
│  5. Symbol cooldown: 30-60min after loss                                       │
│  6. Dynamic conf:    WR<40% → threshold raised                                 │
│  7. Trend filter:    SHORT in uptrend → flip to LONG                          │
│  8. Confidence:      threshold = 0.40 (SHORT), 0.40 (LONG)                   │
│                                                                                 │
│  direction = LONG if pred_pnl > 0 else SHORT                                  │
│  confidence = min(|pred_pnl| / 0.01, 1.0)                                    │
│  vol_mult = 1.0 (fixed for PnL mode)                                          │
└────────────────────────────────────┬────────────────────────────────────────────┘
                                     │ Redis Pub/Sub (orders:signals)
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           OMS EXECUTION (Go 1.22)                              │
│                                                                                 │
│  ┌──────────────────────────────────────────────────────────────────────────┐  │
│  │ ENTRY FILTERS (handleSignal → placeNewEntry): 15 stages                  │  │
│  │                                                                         │  │
│  │  1. Dynamic confidence (OMS-side, cap 0.95)                              │  │
│  │  2. Spread > 0.5% → reject                                              │  │
│  │  3. Zero depth → reject                                                 │  │
│  │  4. OBI momentum ±0.2 against direction → reject                       │  │
│  │  5. Price trend > 0.5% against in 30s → reject                         │  │
│  │  6. Exchange position cross-check                                        │  │
│  │  7. RiskManager: ATR SL, tick filter, EV, Kelly sizing                  │  │
│  └──────────────────────────────────────────────────────────────────────────┘  │
│                                                                                 │
│  ┌──────────────────────────────────────────────────────────────────────────┐  │
│  │ RISK MANAGER (ProcessSignal)                                             │  │
│  │                                                                          │  │
│  │  a. Tick size filter: tick > 0.1% price → reject                        │  │
│  │  b. SL: ATR(14) × 2.0 + wick buffer, clamped to [0.3%, 0.8%]           │  │
│  │  c. EV check: (conf × RR - (1-conf)) ≤ 0 → reject                      │  │
│  │  d. Kelly: half-Kelly, cap 2% risk, vol penalty if volMult > 1.5       │  │
│  └──────────────────────────────────────────────────────────────────────────┘  │
│                                                                                 │
│  ┌──────────────────────────────────────────────────────────────────────────┐  │
│  │ EXIT GRID (BuildExitGrid)                                                │  │
│  │                                                                          │  │
│  │  SL computation:                                                         │  │
│  │    Priority: Dynamic SL > Signal SL > Liquidity SL > ATR SL             │  │
│  │    Range: [0.3%, 0.8%] × sqrt(volMult), min 5 ticks                     │  │
│  │                                                                          │  │
│  │  TP computation (3-tier priority):                                       │  │
│  │    1. Liquidity wall (orderbook support/resistance)                      │  │
│  │    2. ML TP (from Python signal)                                         │  │
│  │    3. Fee-aware formula: entry × (1 + fees + target) / (1 - exit_fee)   │  │
│  │                                                                          │  │
│  │  R:R enforcement: TP ≥ 1.2 × SL distance                                │  │
│  │  Max TP: 3% | Max SL: 0.8% | Min TP: dynamic by price                   │  │
│  └──────────────────────────────────────────────────────────────────────────┘  │
│                                                                                 │
│  ┌──────────────────────────────────────────────────────────────────────────┐  │
│  │ POSITION MANAGEMENT (evaluatePosition)                                   │  │
│  │                                                                          │  │
│  │  1. Hard time-stop:    300s → market close (any remaining qty)           │  │
│  │  2. 180s breakeven:    180s → SL to fillPrice ± 0.13% commission buffer  │  │
│  │  3. monitorExitOrders: TP fills → breakeven → trailing SL               │  │
│  │  4. PositionManager triggers:                                            │  │
│  │     - Time-Stop: 4 candles + R<0.5 + no volume spike → close_full       │  │
│  │     - Scale-Out: R≥1.0 → close 50% at market                           │  │
│  │     - Breakeven: R≥1.5 → SL to entry + 0.15%                           │  │
│  │     - Chandelier: R≥2.0 → trail SL at High/Low ± ATR×2.5               │  │
│  │  5. Queue Monitor: Liquidity Mirage → emergency cancel (wall -75%)      │  │
│  └──────────────────────────────────────────────────────────────────────────┘  │
│                                                                                 │
│  ┌──────────────────────────────────────────────────────────────────────────┐  │
│  │ TRAILING STOP (multi-stage)                                              │  │
│  │                                                                          │  │
│  │  profitR ≥ 4.0 → lock at 3.0R (3× risk locked)                         │  │
│  │  profitR ≥ 2.0 → lock at 1.5R (1.5× risk locked)                       │  │
│  │  profitR ≥ 1.0 → lock at 0.5R (0.5× risk locked)                       │  │
│  │  newSL = mid ± risk × lockR (only tighter, never wider)                 │  │
│  └──────────────────────────────────────────────────────────────────────────┘  │
│                                                                                 │
│  ┌──────────────────────────────────────────────────────────────────────────┐  │
│  │ CONFIDENCE DECAY (decay_exit.go)                                         │  │
│  │                                                                          │  │
│  │  decayDirectionFlipExit: only trails SL in profit (TPs preserved)       │  │
│  │  decayMicrostructureAdverse: only trails SL in profit                   │  │
│  │  NO TP cancellation (fixed from old aggressive behavior)                 │  │
│  └──────────────────────────────────────────────────────────────────────────┘  │
│                                                                                 │
│  ┌──────────────────────────────────────────────────────────────────────────┐  │
│  │ QUEUE MONITOR (queue_monitor.go)                                         │  │
│  │                                                                          │  │
│  │  Tracks passive order wall volumes via WebSocket                         │  │
│  │  Liquidity Mirage: wall evaporated > 75% → emergency cancel             │  │
│  │  Stale order cleanup: cancel after 30s without fill                      │  │
│  └──────────────────────────────────────────────────────────────────────────┘  │
│                                                                                 │
│  ┌──────────────────────────────────────────────────────────────────────────┐  │
│  │ BYBIT V5 API                                                            │  │
│  │  PostOnly Maker entry + Stop-Market SL + PostOnly reduce-only TP        │  │
│  └──────────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Модель: формулы и параметры

### 2.1 Нейросеть

```
Input: ob_seq (2, 60), flow_seq (3, 60), macro (22,), v_memory (8,)
       delta_bars (100, 6) —备用, legacy Conv1d не использует

OrderbookCNN:
  Conv1d(OB_DIM=2 → 16, k=3, pad=1) → ReLU → Conv1d(16 → 32, k=3, pad=1) → ReLU
  → AdaptiveAvgPool1d(1) → FC(32 → EMBED_DIM=32)

FlowGRUAttention:
  GRU(3 → 32, num_layers=1) → SelfAttention(32) → mean(dim=1) → FC(32 → 32)

FusionModel:
  cnn_out(32) + gru_out(32) + macro_proj(22→64) = concat → 128-dim state_vector

DecisionMLP:
  Linear(136 → 64) → ReLU → Dropout(0.45) → Linear(64 → 32) → ReLU → Dropout(0.45)
  → Linear(32 → 3)
  Output: [pred_pnl, trap_logit, toxic_logit]
```

### 2.2 PnL Prediction (текущий режим)

```python
pred_pnl = logits[0]           # Expected PnL (float)
trap_prob = sigmoid(logits[1])  # Trap probability
toxic_prob = sigmoid(logits[2]) # Toxic flow probability

# Direction from sign of pred_pnl
direction = "LONG" if pred_pnl > 0 else "SHORT"

# Confidence = normalized |pred_pnl|
confidence = min(|pred_pnl| / 0.01, 1.0)
```

### 2.3 AsymmetricPnLLoss

```
taker_fee = 0.00075  (0.075%)
maker_fee = 0.00055  (0.055%)

effective_true_pnl = true_pnl - (taker_fee + maker_fee)
error = pred_pnl - effective_true_pnl

weights = 2.5 if error > 0 (overestimation penalized)
        = 1.0 if error ≤ 0 (underestimation)

loss = mean(weights × error²)
```

### 2.4 Delta Bars (50ms time-bins)

```
delta_bid_vol  = current_bid_top10 - previous_bid_top10
delta_ask_vol  = current_ask_top10 - previous_ask_top10
market_buy_vol = Σ(trade.size | side == BUY)  за 50ms
market_sell_vol = Σ(trade.size | side == SELL) за 50ms
trade_count    = count(trades) за 50ms
price_velocity = (mid_now - mid_50ms_ago) / 0.050

Output: (100, 6) — 100 бакетов × 6 фич
```

### 2.5 2D Orderbook Tensor

```
For each of 60 timesteps × 20 levels:
  tensor[t, i, 0] = (price_i - mid) / mid      (relative price, normalized)
  tensor[t, i, 1] = size_i / mean_size_snapshot  (Z-score normalization)

Output: (60, 20, 2) — 60 timestamps × 20 levels × 2 features
```

### 2.6 Feature Vector (22 features)

```
[0]  obi                  = (bid_vol - ask_vol) / (bid_vol + ask_vol)
[1]  cvd_norm             = tanh(cvd / 1e6)
[2]  ofs                  = order_flow_speed = trades / duration
[3]  vwap_dev             = (last_price - vwap) / vwap
[4]  trend_5m             = (price[-1] - price[0]) / price[0] over 300s
[5]  trend_15m            = same over 900s
[6]  trend_1h             = same over 3600s
[7]  trend_4h             = same over 14400s
[8]  trend_1d             = same over 86400s
[9]  funding_rate         = current funding rate
[10] obi_reversal         = max(obi) - min(obi) over 60 ticks
[11] pre_entry_sweep      = 1 if max_trade_size > 3× average in last 10
[12] fill_delay_norm      = avg_delay / 60.0
[13] vol_imbalance        = (buy_vol - sell_vol) / total (from trades)
[14] funding_chg          = obi_last - obi_first (OBI delta as proxy)
[15] depth_imbalance      = (bid_top10 - ask_top10) / total
[16] depth_concentration  = (bid_top3 + ask_top3) / total
[17] spread_bps           = (best_ask - best_bid) / best_bid × 10000
[18] fill_to_depth        = avg_fill / avg_level_size
[19] level_density        = (count_nonzero_bid + count_nonzero_ask) / 20
[20] tanh(bid_vol / 1e6)
[21] tanh(ask_vol / 1e6)
```

---

## 3. SL/TP: формулы и логика

### 3.1 SL Computation (BuildExitGrid)

```
minSLPct = 0.003  (0.3% — minimum SL distance)
maxSLPct = 0.008  (0.8% — hard cap)
slVolMult = sqrt(volatility_multiplier)
minSLPct *= slVolMult
maxSLPct *= slVolMult

SL Priority Chain:
  1. Dynamic SL (from ML Python): slPct clamped to [minSL, maxSL×3]
  2. Signal SL (from ML signal): validated direction, min/max enforced
  3. Liquidity SL: ComputeLiquiditySL — nearest support/resistance zone
  4. Min tick: ≥ 5 ticks from entry

SL for LONG:  fillPrice × (1 - slPct)
SL for SHORT: fillPrice × (1 + slPct)
```

### 3.2 TP Computation (3-tier priority)

```
maxTPDist = MaxTPPct (default 0.015)
if volMult > 1.5: maxTPDist *= 1.5 / volMult

Priority 1: Liquidity Wall
  support/resistance in orderbook
  Must be: feeMinDist ≤ wallDist ≤ maxTPDist
  TP = wall_price ± 2 ticks

Priority 2: ML TP (from Python signal)
  Must be in correct direction, within maxTPDist

Priority 3: Fee-aware formula
  LONG: fillPrice × (1 + entryFee + target) / (1 - exitFee)
  SHORT: fillPrice × (1 - entryFee - target) / (1 + exitFee)
```

### 3.3 R:R Enforcement

```
slDist = |fillPrice - slPrice|
tpDist = |tpPrice - fillPrice|

// Primary check: TP must be ≥ 1.2× SL distance
if tpDist < slDist × 1.2:
    tpPrice = fillPrice ± slDist × 1.2 ± tickSize

// Re-verify after rounding
if tpDistFinal < slDist × 1.0:
    tpPrice = fillPrice ± slDist × 1.2

// Max TP cap: 3% from entry
if tpDist > 0.03:
    tpPrice = fillPrice × (1 ± 0.03)
```

### 3.4 RiskManager ProcessSignal

```
Input: direction, confidence, entryPrice, tpPrice, prices[], balance, volMult, tickSize

Step 1: Tick filter — reject if tickSize / entryPrice > 0.1%

Step 2: ATR-based SL
  atr = CalculateATR(prices, 14)  // Wilder's smoothing
  atrFallback = mean(|price_change|) × 5  // if ATR=0
  extrema = FindNearestExtrema(prices, direction, 20)
  wickBuffer = (max(prices) - min(prices)) × 1.5
  
  LONG SL = entryPrice - atr × 2.0 - wickBuffer
  SHORT SL = entryPrice + atr × 2.0 + wickBuffer
  
  Hard cap: slDistancePct ≤ 0.008 (0.8%)

Step 3: EV check
  rr = tpDistancePct / slDistancePct
  ev = (confidence × rr) - ((1 - confidence) × 1)
  reject if ev ≤ 0

Step 4: Kelly sizing
  kellyPct = confidence - (1 - confidence) / rr
  adjustedKelly = kellyPct × 0.5  (half-Kelly)
  finalRiskPct = min(adjustedKelly, 0.02)  (2% cap)
  if volMult > 1.5: finalRiskPct *= 1.5 / volMult
  
  qty = (balance × finalRiskPct) / slDistancePct
```

### 3.5 Dynamic MinTPPct

```
if mid_price < 0.01:   MinTPPct = 0.01   (1%)
if mid_price < 0.10:   MinTPPct = 0.005  (0.5%)
else:                   MinTPPct = 0.003  (0.3%)
```

---

## 4. Position Management: формулы

### 4.1 PositionManager Triggers

```
CurrentR = unrealizedPnL / OriginalRisk

TRIGGER 1 — Time-Stop:
  if candlesHeld ≥ 4 AND currentR < 0.5 AND NO volume_spike:
    action: close_full

TRIGGER 2 — Scale-Out:
  if currentR ≥ 1.0 AND !ScaledOut:
    action: close_partial (50% of remaining)

TRIGGER 3 — Breakeven:
  if currentR ≥ 1.5 AND !BreakevenSet:
    LONG:  newSL = entryPrice × (1 + 0.0015)
    SHORT: newSL = entryPrice × (1 - 0.0015)
    action: move_sl

TRIGGER 4 — Chandelier Exit:
  if currentR ≥ 2.0:
    atr = CalculateATR(priceHistory, 14)
    LONG:  newSL = candleHigh - atr × 2.5
    SHORT: newSL = candleLow + atr × 2.5
    action: move_sl (tighter only)
```

### 4.2 Hard Time-Stop (300s)

```
holdSec = (now - pos.EntryTime) / 1000
if holdSec > 300:
    side = "Buy" if SHORT, "Sell" if LONG
    PlaceReduceMarket(side, fullRemainingQty)
    // Always checks exchange, never trusts TimeStopPlaced flag
```

### 4.3 180s Breakeven

```
holdSec > 180 AND !BreakevenSet:
    commissionBuffer = fillPrice × 0.0013  (0.13% for both sides)
    LONG:  breakevenPrice = fillPrice + commissionBuffer
    SHORT: breakevenPrice = fillPrice - commissionBuffer
    
    // Only tighten (never widen)
    if newSL tighter than current SL:
        atomicReplaceStopLoss(breakevenPrice, slQty, "breakeven")
        BreakevenSet = true
```

### 4.4 Trailing Stop

```
profitDist = mid - fillPrice (LONG) or fillPrice - mid (SHORT)
risk = |fillPrice - PlannedSL|
profitR = profitDist / risk

if profitR ≥ 4.0: lockR = 3.0
if profitR ≥ 2.0: lockR = 1.5
if profitR ≥ 1.0: lockR = 0.5

LONG:  newSL = mid - risk × lockR
SHORT: newSL = mid + risk × lockR
// Only tighter (never wider)
```

### 4.5 Confidence Decay

```
decayDirectionFlipExit:
  → Only trails SL tighter in profit
  → TPs preserved (NOT cancelled)
  → Grid stays active

decayMicrostructureAdverse:
  → Same behavior: trail SL, preserve grid
```

---

## 5. Фильтры: полная цепочка

### ML Engine (8 фильтров)

| # | Фильтр | Порог | Эффект |
|---|--------|-------|--------|
| 1 | Toxic flow | toxic_prob > 0.40 | HOLD |
| 2 | MIN_EDGE | |pred_pnl| < 0.0025 | HOLD |
| 3 | Pattern memory | 3+ similar losses (cosine ≥ 0.92) | block |
| 4 | Symbol+Setup | 3+ losses avg<-$0.10 | threshold → 0.60 |
| 5 | Symbol cooldown | 30-60min after loss | block |
| 6 | Dynamic confidence | WR < 40% | threshold raised |
| 7 | Trend filter | SHORT in uptrend > 0.3% | flip → LONG |
| 8 | Confidence threshold | SHORT: 0.40, LONG: 0.40 | block |

### OMS (15 фильтров)

| # | Фильтр | Порог | Эффект |
|---|--------|-------|--------|
| 9 | Dynamic confidence (OMS) | cap 0.95, penalty × streak | block |
| 10 | Spread | > 0.5% | reject |
| 11 | Zero depth | total = 0 | reject |
| 12 | OBI momentum | ±0.2 against direction | reject |
| 13 | Price trend | > 0.5% against in 30s | reject |
| 14 | Exchange cross-check | opposite position | skip |
| 15 | RiskManager | tick/SL/EV/Kelly | reject |

### Position Management

| # | Триггер | Условие | Действие |
|---|---------|---------|----------|
| 16 | Hard time-stop | 300s | market close |
| 17 | 180s breakeven | 180s + !BreakevenSet | SL → breakeven |
| 18 | Scale-Out | R ≥ 1.0 | close 50% |
| 19 | Breakeven | R ≥ 1.5 | SL → entry+0.15% |
| 20 | Chandelier | R ≥ 2.0 | trail SL |
| 21 | Queue Monitor | wall -75% | emergency cancel |

---

## 6. Конфигурация

### 6.1 ML Engine (config.py + .env)

| Параметр | Env | Default | Описание |
|----------|-----|---------|----------|
| PREDICT_PNL | PREDICT_PNL | **true** | Режим PnL regression |
| TOXIC_THRESHOLD | TOXIC_THRESHOLD | **0.40** | Порог токсичности |
| MIN_PNL_THRESHOLD | MIN_PNL_THRESHOLD | **0.0025** | Минимальный edge |
| CONFIDENCE_THRESHOLD | CONFIDENCE_THRESHOLD | 0.40 | Base confidence |
| LONG_CONFIDENCE_THRESHOLD | LONG_CONFIDENCE_THRESHOLD | 0.40 | LONG confidence |
| RETRAIN_EPOCHS | RETRAIN_EPOCHS | 12 | Эпохи обучения |
| RETRAIN_TRADE_THRESHOLD | RETRAIN_TRADE_THRESHOLD | 10 | Winning trades → retrain |
| RETRAIN_LOOKBACK_HOURS | RETRAIN_LOOKBACK_HOURS | 24 | Lookback |
| STATE_DIM | STATE_DIM | 128 | State vector size |
| MEMORY_DIM | MEMORY_DIM | 8 | FAISS memory size |
| PATTERN_SIMILARITY | PATTERN_SIMILARITY_THRESHOLD | 0.92 | Cosine threshold |
| PATTERN_TTL | PATTERN_TTL_HOURS | 24 | Pattern expiry |

### 6.2 Go OMS (config.go + .env)

| Параметр | Env | Default | Описание |
|----------|-----|---------|----------|
| MinSLPct | MIN_SL_PCT | 0.006 | Мин SL (overridden by BuildExitGrid: 0.003) |
| MaxSLPct | MAX_SL_PCT | 0.008 | Макс SL (overridden by BuildExitGrid: 0.008) |
| MinTPPct | MIN_TP_PCT | 0.003 | Мин TP (dynamic: 1%/0.5%/0.3% by price) |
| MaxTPPct | MAX_TP_PCT | 0.02 | Макс TP: 2% |
| TradeMarginUSD | TRADE_MARGIN_USD | 10 | Маржа на сделку |
| Leverage | LEVERAGE | 5 | Плечо |
| TimeStopSeconds | TIME_STOP_SECONDS | 3600 | Legacy time stop |
| EntryMakerTicks | ENTRY_MAKER_TICKS | 1 | Ticks for maker entry |

### 6.3 RiskManager Constants

| Константа | Значение | Описание |
|-----------|----------|----------|
| MaxSLPct | 0.008 (0.8%) | Hard SL cap |
| MaxTickSizePct | 0.001 (0.1%) | Reject if tick too large |
| MaxRiskPerTrade | 0.02 (2%) | Kelly risk cap |
| KellyFraction | 0.5 | Half-Kelly |
| ATRMult | 2.0 | ATR multiplier |
| WickBufferMult | 1.5 | Wick buffer multiplier |

### 6.4 PositionManager Constants

| Константа | Значение | Описание |
|-----------|----------|----------|
| ScaleOutR | 1.0 | Scale-out at 1R |
| ScaleOutPct | 0.50 | Close 50% |
| BreakevenR | 1.5 | Breakeven at 1.5R |
| ChandelierR | 2.0 | Chandelier at 2R |
| ChandelierATRMult | 2.5 | ATR multiplier |
| BreakevenFeeBuffer | 0.0015 | 0.15% buffer |
| TimeStopCandles | 4 | Candles for time-stop |

### 6.5 Queue Monitor Constants

| Константа | Значение | Описание |
|-----------|----------|----------|
| LiquidityEvaporationThreshold | 0.25 | Wall -75% → cancel |
| MaxOrderAge | 30s | Stale order cleanup |

---

## 7. Data Flow

```
Bybit WebSocket → Redis (market:orderbook:*)
                      │
                      ├──→ FeatureStore.add_orderbook()
                      │    → 22 macro features
                      │    → 20×60 orderbook tensor
                      │    → 100×6 delta bars
                      │    → 60×3 flow sequence
                      │
                      ├──→ priceHistory[] (rolling 100 mids)
                      └──→ volumeHistory[] (rolling 20 bid volumes)

Model: Fusion(128) → DecisionMLP([pred_pnl, trap, toxic])
           │
           ├──→ direction = LONG/SHORT
           ├──→ confidence = |pred_pnl|/0.01
           └──→ toxic_prob → block if > 0.40
                    │
                    ▼
         OMS Filters (15 stages)
                    │
                    ▼
         RiskManager (ATR SL + Kelly)
                    │
                    ▼
         PostOnly Maker Entry → PendingEntry
                    │
                    ▼
         Fill Monitor (500ms) → ActivePosition
                    │
                    ▼
         Exit Grid Deploy: SL + TP
                    │
                    ▼
         Position Monitor (500ms):
           ├── Hard time-stop (300s)
           ├── 180s breakeven
           ├── TP fills → breakeven → trailing
           ├── PositionManager triggers
           └── Queue Monitor → emergency cancel
```

---

## 8. Git History (PR)

| PR | Описание | Статус |
|----|----------|--------|
| PR #1 | Hard SL cap 0.5% + Conv2d + entry filters | ✅ MERGED |
| PR #2 | Hard time-stop market order | ✅ MERGED |
| PR #3 | Side inversion fix | ✅ MERGED |
| PR #4 | Multimodal Cross-Attention + Queue Toxicity | ✅ MERGED |
| PR #5 | AsymmetricPnLLoss + PnL regression | ✅ MERGED |
| PR #6 | Toxic threshold 0.35→0.50 | ✅ MERGED |
| PR #7 | SL [0.3%-0.8%] + R:R≥1.2 + PnL forced + 180s breakeven + logging | ✅ MERGED |
