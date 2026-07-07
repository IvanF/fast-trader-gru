# Fast Trader GRU — Полная архитектура (финальная, 2026-07-06T22:00Z)

---

## 1. Общая схема

```
Bybit V5 WebSocket (orderbook + trades)
    │
    ▼
┌─────────────────────────────────────────────────────────────────────┐
│  ML Engine (Python 3.11 + ONNX Runtime GPU + FAISS)                 │
│                                                                     │
│  FeatureStore:                                                      │
│    - 50ms Delta Bars (6 фич)                                       │
│    - OrderbookCNN (2×Conv1d → 32-dim)                              │
│    - FlowGRU + SelfAttention (3→32-dim)                            │
│    - 22 macro features (trend, funding, volume imbalance)          │
│                                                                     │
│  FusionModel → 128-dim state_vector                               │
│    CNN(32) + GRU(32) + macro_proj(22→64)                          │
│                                                                     │
│  DecisionMLP → 3 outputs:                                          │
│    [0] pred_pnl (float) — Expected PnL in USD                     │
│    [1] trap_logit → trap_prob (0-1)                                │
│    [2] toxic_logit → toxic_flow_prob (0-1)                         │
│                                                                     │
│  direction = LONG if pred_pnl > 0 else SHORT                       │
│  confidence = min(|pred_pnl| / 0.01, 1.0)                         │
│                                                                     │
│  ╔═══════════════════════════════════════════╗                      │
│  ║  ML FILTERS (8 ступеней)                  ║                      │
│  ╠═══════════════════════════════════════════╣                      │
│  ║ 1. Toxic flow:    toxic_prob > 0.40 → HOLD║                      │
│  ║ 2. MIN_EDGE:      |pred_pnl| < 0.0012    ║                      │
│  ║    → HOLD (0.12% min profit)              ║                      │
│  ║ 3. Pattern Memory: symbol-level, 3+ losses║                     │
│  ║    1h TTL → block                         ║                      │
│  ║ 4. Symbol+Setup:  3+ losses → threshold   ║                      │
│  ║    raised to 0.80                         ║                      │
│  ║ 5. Symbol cooldown: 30-60min after loss   ║                      │
│  ║ 6. Dynamic confidence: WR < 40% → raised  ║                      │
│  ║ 7. Trend filter: SHORT in uptrend → flip  ║                      │
│  ║ 8. Confidence threshold: 0.40             ║                      │
│  ╚═══════════════════════════════════════════╝                      │
└──────────────────────────┬──────────────────────────────────────────┘
                           │ Redis Pub/Sub: orders:signals
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│  OMS Execution (Go 1.22)                                            │
│                                                                     │
│  ╔═══════════════════════════════════════════╗                      │
│  ║  ENTRY FILTERS (16 ступеней)              ║                      │
│  ╠═══════════════════════════════════════════╣                      │
│  ║ 1. Dynamic confidence (cap 0.60)          ║                      │
│  ║ 2. Spread > 0.5% → reject                ║                      │
│  ║ 3. Zero depth → reject                    ║                      │
│  ║ 4. OBI momentum ±0.2 → reject            ║                      │
│  ║ 5. Price trend > 0.5%/30s → reject       ║                      │
│  ║ 6. Volume spike: vol < 2× SMA → reject   ║                      │
│  ║ 7. Exchange cross-check                   ║                      │
│  ║ 8. RiskManager: ATR SL, tick, EV, Kelly   ║                      │
│  ╚═══════════════════════════════════════════╝                      │
│                                                                     │
│  ╔═══════════════════════════════════════════╗                      │
│  ║  RISK MANAGER                             ║                      │
│  ╠═══════════════════════════════════════════╣                      │
│  ║ SL: [0.3%, 0.8%] hard cap, ATR-based     ║                      │
│  ║ Tick filter: tick > 0.1% price → reject   ║                      │
│  ║ EV check: (conf×RR - (1-conf)) ≤ 0 → rej ║                      │
│  ║ Kelly: half-Kelly, cap 2% risk, vol pen   ║                      │
│  ╚═══════════════════════════════════════════╝                      │
│                                                                     │
│  ╔═══════════════════════════════════════════╗                      │
│  ║  EXIT GRID                                ║                      │
│  ╠═══════════════════════════════════════════╣                      │
│  ║ R:R ≥ 1.2 (TP ≥ 1.2 × SL distance)      ║                      │
│  ║ TP priority: liquidity > ML > fee_aware   ║                      │
│  ║ SL priority: dynamic > signal > liq > ATR ║                      │
│  ║ Max TP: 3% | Min TP: dynamic             ║                      │
│  ╚═══════════════════════════════════════════╝                      │
│                                                                     │
│  ╔═══════════════════════════════════════════╗                      │
│  ║  POSITION MANAGEMENT (таймлайн)           ║                      │
│  ╠═══════════════════════════════════════════╣                      │
│  ║ 0s    → Entry                             ║                      │
│  ║ 90s   → Breakeven: SL → fillPrice ± 0.13% ║                      │
│  ║ 180s  → Hard Time-Stop (2-stage FSM):     ║                      │
│  ║          Stage 1: PostOnly Maker (passive) ║                      │
│  ║          Stage 2 (5s): Cancel + Market     ║                      │
│  ║           Kill-Switch! (guaranteed exit)   ║                      │
│  ╠═══════════════════════════════════════════╣                      │
│  ║ Scale-Out: 50% at 1R profit              ║                      │
│  ║ Breakeven: SL → entry at 1.5R             ║                      │
│  ║ Chandelier: ATR trail at 2R               ║                      │
│  ║ Trailing SL: 1R→0.5R, 2R→1.5R, 4R→3R     ║                      │
│  ╚═══════════════════════════════════════════╝                      │
│                                                                     │
│  ╔═══════════════════════════════════════════╗                      │
│  ║  CLOSE REASON RESOLUTION                  ║                      │
│  ╠═══════════════════════════════════════════╣                      │
│  ║ if TimeStopPlaced → "time_stop"           ║                      │
│  ║ if Exit ≈ TP → "take_profit"              ║                      │
│  ║ if Exit ≈ SL → "stop_loss"               ║                      │
│  ║ else → "exchange_closed"                  ║                      │
│  ╚═══════════════════════════════════════════╝                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 2. ML Pipeline

```
Bybit WebSocket → Redis (orderbook + trades)
         │
         ▼
FeatureStore:
  add_orderbook() → 22 macro features + 20×60 orderbook tensor
  add_trade()     → 60×3 flow sequence + 50ms delta bars
         │
         ▼
Inference: ONNX Runtime (CUDA)
  Conv2d CNN (2×Conv1d → 32-dim)
  FlowGRU + SelfAttention (3→32-dim)
  FusionModel: CNN(32) + GRU(32) + macro(22→64) → 128-dim state_vector
  DecisionMLP: shared(128→64) → head(64→3)
         │
         ▼
Output: pred_pnl, trap_prob, toxic_flow_prob
         │
         ▼
direction = LONG if pred_pnl > 0 else SHORT
confidence = min(|pred_pnl| / 0.01, 1.0)
         │
         ▼
ML Filters (8) → Redis Pub/Sub → OMS Filters (16) → RiskManager → Bybit
```

---

## 3. Exit Pipeline (подробно)

```
Position opened → Exit Grid Deployed (SL + TP)
         │
         ▼
evaluatePosition() — 500ms tick:
  │
  ├─ syncPositionFromExchange() → verify exchange state
  │
  ├─ monitorExitOrders():
  │   ├─ TP fills → breakeven SL → trailing SL
  │   ├─ SL fills → finalize
  │   └─ check fill status
  │
  ├─ retryMissingTakeProfits()
  │
  ├─ HARD TIME-STOP (holdSec > 180):
  │   ├─ cancelExitOrders()
  │   ├─ PlaceReducePostOnlyLimit (Stage 1)
  │   │   └─ goWithTimeout(5s):
  │   │       ├─ mu.Lock() → check positions[Symbol]
  │   │       ├─ GetOrderRealtime()
  │   │       ├─ if "New" → CancelOrder()
  │   │       └─ PlaceReduceMarketRetry (Stage 2: Kill-Switch!)
  │   └─ if PostOnly failed → immediate market
  │
  ├─ monitorExitOrders() (second pass)
  │
  ├─ BREAKEVEN (holdSec > 90):
  │   ├─ commissionBuffer = FillPrice × 0.0013
  │   ├─ LONG: breakevenPrice = FillPrice + buffer
  │   ├─ SHORT: breakevenPrice = FillPrice - buffer
  │   └─ atomicReplaceStopLoss (only tighter)
  │
  └─ PositionManager:
      ├─ Scale-Out: 50% at 1R
      ├─ Breakeven: SL → entry at 1.5R
      ├─ Chandelier: ATR trail at 2R
      ├─ Time-Stop (4 candles + R < 0.5): → timeStopLimitExit()
      │   └─ PostOnly + 5s Kill-Switch (same FSM)
      └─ Queue Monitor: wall evaporation → cancel
```

---

## 4. SL/TP Формулы

### SL Computation
```
minSLPct = 0.3%, maxSLPct = 0.8%
slVolMult = sqrt(volatilityMultiplier)
minSLPct *= slVolMult, maxSLPct *= slVolMult

SL Priority:
  1. Dynamic SL (from ML Python signal)
  2. Signal SL (from ML signal)
  3. Liquidity SL (orderbook zones)
  4. Min tick: ≥ 5 ticks from entry

Clamp: SL = clamp(SL, min=fillPrice×(1-dirs×0.003), max=fillPrice×(1-dirs×0.008))
```

### TP Computation
```
R:R enforcement: TP ≥ 1.2 × SL distance
Max TP: 3%
Min TP: dynamic (1% for <$0.01 coins, 0.5% for <$0.10)

TP Priority:
  1. Liquidity wall (orderbook support/resistance)
  2. ML TP (from Python signal)
  3. Fee-aware: fill × (1 + fees + target) / (1 - exit_fee)
```

### Trailing Stop
```
profitR ≥ 4.0 → lock at 3.0R
profitR ≥ 2.0 → lock at 1.5R
profitR ≥ 1.0 → lock at 0.5R
newSL = mid ± risk × lockR (only tighter, never wider)
```

---

## 5. Конфигурация

| Параметр | Значение | Файл |
|----------|----------|------|
| PREDICT_PNL | true | docker-compose.yml |
| MIN_EDGE | 0.0012 | engine.py:1155 |
| MIN_PNL_THRESHOLD | 0.0012 | config.py:110, docker-compose.yml |
| TOXIC_THRESHOLD | 0.40 | engine.py |
| CONFIDENCE_THRESHOLD | 0.40 | .env |
| Effective Confidence Cap | 0.60 | service.go:93 |
| Pattern Memory TTL | 1 hour | online_learner.py:33 |
| PATTERN_SIMILARITY | 0.92 | .env |
| PATTERN_BLOCK_LIMIT | 50 | engine.py |
| Escalation threshold | 0.80 | engine.py |
| Hard Time-Stop | 180s (3 min) | service.go:1315 |
| Breakeven | 90s (1.5 min) | service.go:1397 |
| Kill-Switch timeout | 5s | service.go:1316, exits.go:824 |
| SL range | [0.3%, 0.8%] | exit_grid.go |
| R:R enforcement | ≥ 1.2 | exit_grid.go:284 |
| Scale-out | 50% at 1R | position_manager.go:14 |
| Max TP | 3% | exit_grid.go |
| RETRAIN_TRADE_THRESHOLD | 50 | docker-compose.yml |
| RETRAIN_EPOCHS | 12 | docker-compose.yml |
| Entry Maker Ticks | config | .env |
| PositionManager Time-Stop | 4 candles, R < 0.5 | position_manager.go:94 |

---

## 6. Kill-Switch FSM (детально)

```
holdSec > 180:
  │
  ├── pos.TimeStopPlaced == true?
  │   └── YES → check if already closed → finalize
  │
  └── NO → initiate 2-stage exit:
      │
      ├── cancelExitOrders (cancel existing SL/TP)
      │
      ├── Stage 1: PlaceReducePostOnlyLimit
      │   ├── SUCCESS → pos.TimeStopPlaced = true
      │   │   └── goWithTimeout(5s):
      │   │       ├── mu.Lock()
      │   │       ├── check positions[Symbol].TimeStopPlaced
      │   │       ├── mu.Unlock()
      │   │       ├── GetOrderRealtime(orderID)
      │   │       ├── if orderStatus == "New":
      │   │       │   ├── CancelOrder(orderID)
      │   │       │   └── PlaceReduceMarketRetry → "Market Kill-Switch!"
      │   │       └── (else: order filled, nothing to do)
      │   └── return
      │
      └── FAILED → immediate PlaceReduceMarketRetry → "Market Kill-Switch!"
```

---

## 7. Статистика

### Backtest (190 сделок)
| Конфигурация | PnL | R:R | EV |
|-------------|-----|-----|-----|
| Old (SL 1.5%, R:R 0.7, 300s) | -$9.87 | 0.66 | -$0.052 |
| SL 0.5% + R:R 1.2 | -$3.35 | 0.87 | -$0.018 |
| + 120s passive Maker | -$3.06 | 0.89 | -$0.016 |
| + escalation 0.80 | **-$1.51** | **0.96** | **-$0.008** |

### Live (24h после всех правок)
| Метрика | Значение |
|---------|----------|
| Trades | 5 (1W/4L) |
| PnL | -$0.17 |
| WR | 20% |
| R:R | 0.13 |
| Balance | $1,999.83 |
| Kill-Switch executions | 3 (XAUTUSDT, ETHUSDT, ZECUSDT) |
| FAISS entries | 2,772 |
| ML pass rate | 17.4% (MIN_EDGE=0.0012) |

---

## 8. Все PR за проект

| # | Описание | Статус |
|---|----------|:------:|
| PR #1 | Hard SL + Conv2d + filters | ✅ |
| PR #2 | Hard time-stop market order | ✅ |
| PR #3 | Side inversion fix | ✅ |
| PR #4 | Cross-Attention + Queue Toxicity | ✅ |
| PR #5 | AsymmetricPnLLoss + PnL regression | ✅ |
| PR #6 | Toxic threshold 0.35→0.50 | ✅ |
| PR #7 | SL [0.3%-0.8%] + R:R≥1.2 + 180s breakeven | ✅ |
| PR #8 | Volume spike filter + time-stop reason | ✅ |
| PR #9 | Kill-Switch FSM + timings + MIN_EDGE 0.0012 | ✅ |
| PR #10 | Kill-Switch in timeStopLimitExit + indentation | ✅ |

---

## 9. Известные ограничения

1. **WR 20% (5 сделок)** — слишком мало данных для оценки. Нужно 100+ сделок.
2. **pred_pnl не нормализован** — модель предсказывает $1-$84 PnL на $10 марже.
3. **7 символов доминируют в убытках** — VANRYUSDT, HMSTRUSDT и др.
4. **Conv2d ONNX не переобучен** — legacy Conv1d используется в runtime.
5. **Queue Monitor не подключён к service.go** — код написан, тесты проходят, но не wired.
