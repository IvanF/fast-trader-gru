# Fast Trader GRU — Текущая архитектура (финальная, 2026-07-06)

---

## 1. Общая схема

```
Bybit V5 WebSocket (данные)
    │
    ▼
┌─────────────────────────────────────────────────────────────────────┐
│  ML Engine (Python 3.10 + ONNX GPU)                                │
│                                                                     │
│  FeatureStore:                                                      │
│    - 50ms Delta Bars (6 фич)                                       │
│    - OrderbookCNN (2×Conv1d → 32-dim)                              │
│    - FlowGRU + Attention (3→32-dim)                                │
│    - 22 macro features (trend, funding, volume imbalance)          │
│                                                                     │
│  FusionModel → 128-dim state_vector                               │
│    CNN(32) + GRU(32) + macro_proj(22→64)                          │
│                                                                     │
│  DecisionMLP → 3 outputs:                                          │
│    [0] pred_pnl (float) — Expected PnL                             │
│    [1] trap_logit → trap_prob                                      │
│    [2] toxic_logit → toxic_flow_prob                               │
│                                                                     │
│  direction = LONG if pred_pnl > 0 else SHORT                       │
│  confidence = min(|pred_pnl| / 0.01, 1.0)                         │
│                                                                     │
│  ФИЛЬТРЫ ML (8):                                                    │
│    1. Toxic flow: toxic_prob > 0.40 → HOLD                         │
│    2. MIN_EDGE: |pred_pnl| < 0.0005 → HOLD                         │
│    3. Pattern Memory: symbol-level, 3+ losses, 1h TTL             │
│    4. Symbol+Setup: 3+ losses → threshold 0.80                     │
│    5. Symbol cooldown: 30-60min after loss                          │
│    6. Dynamic confidence: WR < 40% → raised                        │
│    7. Trend filter: SHORT in uptrend → flip to LONG                │
│    8. Confidence threshold: 0.40                                   │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│  OMS Execution (Go 1.22)                                            │
│                                                                     │
│  ENTRY FILTERS (16):                                                │
│    1. Dynamic confidence (OMS-side, cap 0.60)                       │
│    2. Spread > 0.5% → reject                                       │
│    3. Zero depth → reject                                          │
│    4. OBI momentum ±0.2 → reject                                   │
│    5. Price trend > 0.5%/30s → reject                              │
│    6. Volume spike: volume < 2× SMA → reject                       │
│    7. Exchange cross-check                                          │
│    8. RiskManager: ATR SL, tick filter, EV, Kelly                  │
│                                                                     │
│  RISK MANAGER:                                                      │
│    - SL: [0.3%, 0.8%] hard cap, ATR-based                         │
│    - Tick filter: tick > 0.1% price → reject                       │
│    - EV check: (conf×RR - (1-conf)) ≤ 0 → reject                 │
│    - Kelly: half-Kelly, cap 2% risk, vol penalty                    │
│                                                                     │
│  EXIT GRID:                                                         │
│    - R:R ≥ 1.2 (TP ≥ 1.2× SL)                                    │
│    - TP priority: liquidity > ML > fee_aware                       │
│    - SL priority: dynamic > signal > liquidity > ATR               │
│    - Max TP: 3% | Min TP: dynamic (1%/<$0.01, 0.5%/<$0.10)      │
│                                                                     │
│  POSITION MANAGEMENT:                                               │
│    - Hard time-stop: 120s → passive Maker close                    │
│    - 180s breakeven: SL → fillPrice ± 0.13%                        │
│    - Scale-Out: 50% at 1R profit                                   │
│    - Breakeven: SL → entry at 1.5R                                 │
│    - Chandelier: ATR trail at 2R                                   │
│    - Trailing SL: 1R→0.5R, 2R→1.5R, 4R→3R lock                   │
│    - Queue Monitor: Liquidity Mirage detection                     │
│                                                                     │
│  CONFIDENCE DECAY:                                                  │
│    - Only trails SL tighter (TPs preserved, NO cancel)             │
│                                                                     │
│  CLOSE REASON:                                                      │
│    - time_stop (hard 120s)                                         │
│    - take_profit                                                    │
│    - stop_loss                                                      │
│    - stale_tracker_removed                                         │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 2. ML Pipeline

```
Data: Bybit WebSocket → Redis (orderbook + trades)
     ↓
FeatureStore:
  add_orderbook() → 22 macro features + 20×60 orderbook tensor
  add_trade() → 60×3 flow sequence + 50ms delta bars
     ↓
Inference: ONNX (Conv2d CNN + FlowGRU + DecisionMLP)
     ↓
output: pred_pnl, trap_prob, toxic_flow_prob
     ↓
direction = LONG if pred_pnl > 0 else SHORT
confidence = min(|pred_pnl| / 0.01, 1.0)
     ↓
ML Filters (8) → OMS Filters (16) → RiskManager → Order
```

---

## 3. Exit Pipeline

```
Position opened → Exit Grid Deployed (SL + TP)
     ↓
monitorExitOrders (500ms):
  - TP fills → breakeven SL → trailing SL
  - SL fills → finalize
  - 120s hard time-stop → passive Maker close (PostOnly + 8s fallback)
  - 180s breakeven → SL to fillPrice ± 0.13%
     ↓
PositionManager (500ms):
  - Scale-Out: 50% at 1R
  - Breakeven: SL → entry at 1.5R
  - Chandelier: ATR trail at 2R
  - Queue Monitor: wall evaporation → cancel
     ↓
Close reason resolution:
  - TimeStopPlaced → "time_stop"
  - Exit ≈ TP → "take_profit"
  - Exit ≈ SL → "stop_loss"
  - else → "exchange_closed"
```

---

## 4. SL/TP Формулы

### SL Computation
```
minSLPct = 0.3%, maxSLPct = 0.8%
slVolMult = sqrt(volatilityMultiplier)
minSLPct *= slVolMult, maxSLPct *= slVolMult

SL Priority:
  1. Dynamic SL (from ML Python)
  2. Signal SL (from ML signal)
  3. Liquidity SL (orderbook zones)
  4. Min tick: ≥ 5 ticks from entry
```

### TP Computation
```
R:R enforcement: TP ≥ 1.2 × SL distance
Max TP: 3% | Min TP: dynamic (1%/$0.01, 0.5%/$0.10)

TP Priority:
  1. Liquidity wall (orderbook support/resistance)
  2. ML TP (from Python signal)
  3. Fee-aware formula: fill × (1 + fees + target) / (1 - exit_fee)
```

### Trailing Stop
```
profitR ≥ 4.0 → lock at 3.0R
profitR ≥ 2.0 → lock at 1.5R
profitR ≥ 1.0 → lock at 0.5R
newSL = mid ± risk × lockR (only tighter)
```

---

## 5. Конфигурация

| Параметр | Значение |
|----------|----------|
| PREDICT_PNL | true |
| MIN_EDGE | 0.0005 |
| TOXIC_THRESHOLD | 0.40 |
| CONFIDENCE_THRESHOLD | 0.40 |
| Effective Confidence Cap | 0.60 |
| Pattern Memory TTL | 1 hour |
| PATTERN_SIMILARITY | 0.92 |
| Escalation threshold | 0.80 |
| Hard Time-Stop | 120s |
| Breakeven | 180s |
| SL range | [0.3%, 0.8%] |
| R:R enforcement | ≥ 1.2 |
| Scale-out | 50% at 1R |
| Max TP | 3% |
| RETRAIN_TRADE_THRESHOLD | 50 |
| RETRAIN_EPOCHS | 12 |

---

## 6. Статистика (backtest 190 сделок)

| Конфигурация | PnL | R:R | EV |
|-------------|-----|-----|-----|
| Old (SL 1.5%, R:R 0.7, 300s) | -$9.87 | 0.66 | -$0.052 |
| SL 0.5% + R:R 1.2 | -$3.35 | 0.87 | -$0.018 |
| + 120s passive Maker | -$3.06 | 0.89 | -$0.016 |
| + escalation 0.80 | **-$1.51** | **0.96** | **-$0.008** |

**Улучшение**: PnL +$8.36, R:R +0.30, EV +$0.044
**Нужно WR 51%** для безубыточности (сейчас 49%)
