# Fast Trader GRU — Полная архитектура (2026-07-07)
# Актуальная версия после 12 PR

---

## 1. Общая архитектура

```
┌─────────────────────────────────────────────────────────────────────┐
│                     Bybit V5 WebSocket API                         │
│              orderbook (L2) + trades + positions + orders          │
└──────────┬──────────────────────────────────┬───────────────────────┘
           │                                  │
           ▼                                  ▼
┌─────────────────────┐              ┌─────────────────────────────┐
│   ML Engine         │              │   OMS Execution (Go)        │
│   Python 3.11       │              │   16 entry filters          │
│   ONNX Runtime GPU  │              │   Risk Manager              │
│   FAISS (2772)      │              │   Exit Grid                 │
│   Retrain: 50 trades│              │   Position Manager          │
│                     │              │   Kill-Switch FSM           │
│   595K preds →      │   Redis      │   2-stage time-stop        │
│   129K signals ─────│──────────►   │   Breakeven 90s            │
│                     │   Pub/Sub    │   Trailing SL               │
│                     │              │                             │
│   Pass rate: 21.8%  │              │   Balance: $1,985.28        │
└─────────────────────┘              └─────────────────────────────┘
```

---

## 2. ML Pipeline

### 2.1 Data Flow

```
Bybit WebSocket
    │
    ├─ orderbook (L2) ──► Redis ──► add_orderbook()
    │                                  │
    │                                  ├─ 22 macro features (trend, funding, imbalance)
    │                                  └─ 20×60 orderbook tensor
    │
    └─ trades ──────────► Redis ──► add_trade()
                                     │
                                     ├─ 60×3 flow sequence
                                     └─ 50ms delta bars (6 features)
```

### 2.2 Feature Engineering

```
FeatureStore → FusionModel → DecisionMLP → Signal

┌─────────────────────────────────────────────────────────────┐
│  FusionModel (128-dim state_vector)                         │
│                                                             │
│  Input 1: OrderbookCNN                                      │
│    20 depth levels × 60 timesteps × 2 channels             │
│    Channel 0: relative_price = (price - mid) / mid          │
│    Channel 1: z_normalized_volume = size / mean_size        │
│    Conv1d(2→16, k=3) → Conv1d(16→32, k=3) → FC(32)        │
│    Output: 32-dim                                            │
│                                                             │
│  Input 2: FlowGRU + SelfAttention                           │
│    60 timesteps × 3 features (delta_bid, delta_ask, trade)  │
│    GRU(3→32, 2 layers) → SelfAttention                     │
│    Output: 32-dim                                            │
│                                                             │
│  Input 3: Macro Features                                    │
│    22 features: trend_5m, trend_15m, funding_rate,          │
│    volume_imbalance, volatility, regime, btc_correlation...  │
│    Linear(22→64)                                             │
│    Output: 64-dim                                            │
│                                                             │
│  Concat: CNN(32) + GRU(32) + macro(64) = 128-dim           │
└─────────────────────────────────────────────────────────────┘
```

### 2.3 DecisionMLP (PnL Regression Mode)

```
state_vector (128)
    │
    ▼
DecisionMLP:
  shared: Linear(128→64) → ReLU → Dropout(0.3)
  head:   Linear(64→3)

Output 3 vectors:
  [0] pred_pnl     → Expected PnL in USD (float)
  [1] trap_logit   → trap_prob via sigmoid (0-1)
  [2] toxic_logit  → toxic_flow_prob via sigmoid (0-1)

Training: AsymmetricPnLLoss (2.5× overestimation penalty)
effective_true_pnl = pnl - (taker 0.075% + maker 0.055%)
```

### 2.4 Signal Generation

```
direction  = LONG if pred_pnl > 0 else SHORT
confidence = min(|pred_pnl| / 0.01, 1.0)
```

### 2.5 ML Filters (8 ступеней)

| # | Фильтр | Условие | Действие |
|---|--------|---------|----------|
| 1 | Toxic flow | toxic_prob > 0.40 | → HOLD |
| 2 | MIN_EDGE | \|pred_pnl\| < 0.0012 | → HOLD |
| 3 | Pattern Memory | symbol-level, 3+ losses, 1h TTL | → block |
| 4 | Symbol+Setup | 3+ losses → threshold 0.80 | → raise bar |
| 5 | Symbol cooldown | 30-60min after loss | → block |
| 6 | Dynamic confidence | WR < 40% → raised | → raise bar |
| 7 | Trend filter | SHORT in uptrend | → flip to LONG |
| 8 | Confidence threshold | confidence < 0.40 | → HOLD |

---

## 3. OMS Execution Pipeline

### 3.1 Signal Reception (Redis Pub/Sub)

```
ML Engine ──► PUBLISH orders:signals ──► OMS SUBSCRIBE

Signal JSON:
{
  "symbol": "HYPEUSDT",
  "direction": "LONG",
  "confidence": 0.44,
  "dynamic_sl_pct": 0.004,
  "dynamic_tp_pct": 0.014,
  "entry_price": 71.458,
  "stop_loss": 71.172,
  "take_profits": [72.478]
}
```

### 3.2 Entry Filters (16 ступеней)

```
Signal received
    │
    ├─ 1. Dynamic Confidence: confidence < DynamicConfCap(0.45) → reject
    │     effectiveConf = baseConf × Penalty()
    │     Penalty() = max(1.0, 3.0 - 2.5 × (wr / 0.40))
    │     streak_bonus = 1 + (consec - 1) × 0.3
    │     cap: 0.45
    │
    ├─ 2. Spread: spread > 0.5% → reject
    │     spread = (ask - bid) / mid
    │
    ├─ 3. Zero depth: bid_depth = 0 OR ask_depth = 0 → reject
    │
    ├─ 4. OBI momentum: |obi| > 0.4 → reject
    │     obi = (bid_vol - ask_vol) / (bid_vol + ask_vol)
    │     LONG rejected if obi > 0.4 (buying pressure → price up → already late)
    │     SHORT rejected if obi < -0.4
    │
    ├─ 5. Price trend: move > 0.5% in 30s → reject
    │     move = (price_now - price_30s_ago) / price_30s_ago
    │
    ├─ 6. Volume: current_vol < 0.5 × SMA(20 ticks) → reject
    │     current_vol = sum of top-10 bid sizes
    │     SMA = average of last 20 volume snapshots
    │
    ├─ 7. Exchange cross-check
    │
    └─ 8. RiskManager (see §3.3)
```

### 3.3 Risk Manager

```
┌─────────────────────────────────────────────────────────┐
│  RISK MANAGER                                           │
│                                                         │
│  SL Computation:                                        │
│    ATR(14) × ATRMult(2.0) = raw SL distance            │
│    minSLPct = 0.3% × sqrt(volatilityMultiplier)        │
│    maxSLPct = 0.8% × sqrt(volatilityMultiplier)        │
│    SL = clamp(raw, min=minSLPct, max=maxSLPct)          │
│                                                         │
│  SL Priority:                                           │
│    1. Dynamic SL (from ML signal)                       │
│    2. Signal SL (from ML signal)                        │
│    3. Liquidity SL (orderbook zones)                    │
│    4. Min tick: ≥ 5 ticks from entry                    │
│                                                         │
│  TP Computation:                                        │
│    R:R enforcement: TP ≥ 1.2 × SL distance             │
│    Max TP: 3%                                           │
│    Min TP: dynamic (1%/<$0.01, 0.5%/<$0.10)           │
│                                                         │
│  TP Priority:                                           │
│    1. Liquidity wall (orderbook support/resistance)     │
│    2. ML TP (from Python signal)                        │
│    3. Fee-aware: fill × (1 + fees + target) / (1 - exit)│
│                                                         │
│  Additional checks:                                     │
│    - Tick filter: tick > 0.1% price → reject            │
│    - EV check: (conf × RR - (1-conf)) ≤ 0 → reject    │
│    - Kelly: half-Kelly, cap 2% risk, vol penalty        │
└─────────────────────────────────────────────────────────┘
```

---

## 4. Position Timeline (таймлайн сделки)

```
0s ──────── Entry (PostOnly Maker)
│
│  monitorExitOrders() every 500ms
│
├─ TP fills → partial breakeven SL → trailing SL
├─ SL fills → finalize
│
├─ 90s ──── BREAKEVEN
│            SL → fillPrice ± 0.13% (commissionBuffer)
│            Only tightens (never widens)
│            Formula: breakevenPrice = fillPrice ± fillPrice × 0.0013
│
├─ 180s ─── HARD TIME-STOP (2-stage FSM)
│            │
│            ├─ Stage 1: PostOnly Maker exit
│            │   exitPrice = PassiveMakerExitPrice(direction, ob, tickSize, ticks)
│            │   PlaceReducePostOnlyLimit(symbol, side, qty, price)
│            │   pos.TimeStopPlaced = true
│            │
│            └─ Stage 2 (30s goroutine timeout):
│                if pos still open:
│                  CancelOrder(orderID)
│                  PlaceReduceMarketRetry() → "Market Kill-Switch!"
│
├─ 300s ─── ZOMBIE RETRY (if Kill-Switch failed)
│            reset TimeStopPlaced = false
│            PlaceReduceMarketRetry() → "ZOMBIE force close"
│
└─ Close reason resolution:
     TimeStopPlaced → "time_stop"
     Exit ≈ TP → "take_profit"
     Exit ≈ SL → "stop_loss"
     else → "exchange_closed"
```

---

## 5. Exit Grid

### 5.1 Trailing Stop

```
profitR ≥ 4.0 → lock at 3.0R
profitR ≥ 2.0 → lock at 1.5R
profitR ≥ 1.0 → lock at 0.5R

newSL = mid ± risk × lockR (only tighter, never wider)
```

### 5.2 Scale-Out

```
At 1R profit: close 50% of position (ScaleOutPct = 0.50)
Remainder: trail with trailing SL
```

### 5.3 Chandelier Exit

```
At 2R profit: trail ATR-based stop
chandelierSL = high - ATR(14) × 2.0 (for LONG)
chandelierSL = low + ATR(14) × 2.0 (for SHORT)
```

---

## 6. Паттерн Мемори (FAISS)

```
state_vector (128-dim) → FAISS index (cosine similarity)

Block conditions:
  - cosine_similarity ≥ 0.92
  - ≥ 3 similar patterns with avg_loss < -$0.10
  - Symbol-level: losses counted per symbol (not cross-symbol)
  - TTL: 1 hour (auto-expire old patterns)

Escalation:
  - 3+ losses in same symbol+regime+direction
  - threshold raised to 0.80

Cooldown:
  - 30min after 1 loss
  - 60min after 2+ consecutive losses
```

---

## 7. Конфигурация (все параметры)

| Параметр | Значение | Описание |
|----------|----------|----------|
| PREDICT_PNL | true | PnL regression mode |
| MIN_EDGE | 0.0012 | Минимальный pred_pnl (0.12%) |
| TOXIC_THRESHOLD | 0.40 | Toxic flow probability |
| CONFIDENCE_THRESHOLD | 0.40 | Базовый порог confidence |
| DynamicConfCap | 0.45 | Хард-кап confidence |
| Pattern Memory TTL | 1 hour | Время жизни паттерна |
| PATTERN_SIMILARITY | 0.92 | Косинусное сходство |
| PATTERN_BLOCK_LIMIT | 50 | Макс. паттернов в блоке |
| Escalation threshold | 0.80 | Порог для symbol+setup |
| Hard Time-Stop | 180s | Жёсткий выход |
| Kill-Switch timeout | 30s | PostOnly → Market |
| Zombie Retry | 120s | Retry если Kill-Switch failed |
| Breakeven | 90s | SL → fillPrice |
| SL range | [0.3%, 0.8%] | ATR-based, clamped |
| R:R enforcement | ≥ 1.2 | Min TP/SL ratio |
| Scale-out | 50% at 1R | Частичное закрытие |
| Max TP | 3% | Макс. Take Profit |
| RETRAIN_TRADE_THRESHOLD | 50 | Trades before retrain |
| RETRAIN_EPOCHS | 12 | Epochs per retrain |
| Entry Maker Ticks | config | PostOnly ticks |
| PM Time-Stop | 4 candles, R < 0.5 | PositionManager trigger |
| Volume filter | 0.5× SMA | Reject dead markets |
| OBI momentum | ±0.4 | Reject strong imbalance |
| Price trend | 0.5%/30s | Reject momentum |

---

## 8. Инфраструктура

| Сервис | Технология | Порт |
|--------|-----------|------|
| ML Engine | Python 3.11 + ONNX GPU | internal |
| OMS Execution | Go 1.22 | internal |
| Redis | Redis 7 | :16379 |
| InfluxDB | InfluxDB 2 | :18086 |
| Grafana | Grafana 10 | :13000 |
| Prometheus | Prometheus | :19090 |

---

## 9. Git History (12 PR)

| # | Описание | Статус |
|---|----------|:------:|
| PR #1 | Hard SL + Conv2d + filters | ✅ |
| PR #2 | Hard time-stop market order | ✅ |
| PR #3 | Side inversion fix | ✅ |
| PR #4 | Cross-Attention + Queue Toxicity | ✅ |
| PR #5 | AsymmetricPnLLoss + PnL regression | ✅ |
| PR #6 | Toxic threshold 0.35→0.50 | ✅ |
| PR #7 | SL [0.3%-0.8%] + R:R≥1.2 + breakeven | ✅ |
| PR #8 | Volume spike filter + time-stop reason | ✅ |
| PR #9 | Kill-Switch FSM + timings + MIN_EDGE | ✅ |
| PR #10 | Kill-Switch in timeStopLimitExit | ✅ |
| PR #11 | Volume 0.5x + OBI ±0.4 + zombie retry | ✅ |
| PR #12 | DynamicConfCap 0.60→0.45 | ✅ |
