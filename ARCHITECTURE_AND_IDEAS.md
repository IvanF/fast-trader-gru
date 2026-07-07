# Fast Trader GRU — Текущая архитектура и идеи по тестированию
# Дата: 2026-07-06

---

## 1. Архитектура системы

### 1.1 Общая схема

```
Bybit V5 WebSocket (данные)
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│  ML Engine (Python 3.10 + ONNX GPU)                         │
│                                                             │
│  FeatureStore:                                               │
│    - 50ms Delta Bars (6 фич: delta_bid/ask_vol,             │
│      market_buy/sell_vol, trade_count, price_velocity)      │
│    - OrderbookCNN (2×Conv1d → 32-dim)                        │
│    - FlowGRU + SelfAttention (3→32-dim)                      │
│    - 22 macro features (trend, funding, volume imbalance)    │
│                                                             │
│  FusionModel → 128-dim state_vector                         │
│    CNN(32) + GRU(32) + macro_proj(22→64)                    │
│                                                             │
│  DecisionMLP → 3 outputs:                                    │
│    [0] pred_pnl (float) — Expected PnL                      │
│    [1] trap_logit → trap_prob (0-1)                          │
│    [2] toxic_logit → toxic_flow_prob (0-1)                   │
│                                                             │
│  direction = LONG if pred_pnl > 0 else SHORT                │
│  confidence = min(|pred_pnl| / 0.01, 1.0)                   │
│                                                             │
│  ФИЛЬТРЫ ML:                                                │
│    1. Toxic flow: toxic_prob > 0.40 → HOLD                  │
│    2. MIN_EDGE: |pred_pnl| < 0.0005 → HOLD                 │
│    3. Pattern Memory: symbol-level, 3+ losses, 1h TTL      │
│    4. Symbol+Setup: 3+ losses → threshold 0.80              │
│    5. Symbol cooldown: 30-60min after loss                   │
│    6. Dynamic confidence: WR < 40% → raised                  │
│    7. Trend filter: SHORT in uptrend → flip to LONG          │
│    8. Confidence threshold: 0.40                              │
└───────────────────────────┬──────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  OMS Execution (Go)                                         │
│                                                             │
│  ENTRY FILTERS (15):                                        │
│    1. Dynamic confidence (OMS-side, cap 0.60)               │
│    2. Spread > 0.5% → reject                                │
│    3. Zero depth → reject                                   │
│    4. OBI momentum ±0.2 → reject                            │
│    5. Price trend > 0.5%/30s → reject                       │
│    6. Exchange cross-check                                   │
│    7. RiskManager: ATR SL, tick filter, EV, Kelly            │
│                                                             │
│  RISK MANAGER:                                              │
│    - SL: [0.3%, 0.8%] hard cap, ATR-based                  │
│    - Tick filter: tick > 0.1% price → reject                │
│    - EV check: (conf×RR - (1-conf)) ≤ 0 → reject           │
│    - Kelly: half-Kelly, cap 2% risk, vol penalty             │
│                                                             │
│  EXIT GRID:                                                 │
│    - R:R ≥ 1.2 (TP ≥ 1.2× SL distance)                    │
│    - TP priority: liquidity > ML > fee_aware                │
│    - SL priority: dynamic > signal > liquidity > ATR        │
│    - Max TP: 3% | Min TP: dynamic (1%/<$0.01, 0.5%/<$0.10)│
│                                                             │
│  POSITION MANAGEMENT:                                       │
│    - Hard time-stop: 120s → passive Maker close             │
│    - 180s breakeven: SL → fillPrice ± 0.13%                 │
│    - Scale-Out: 50% at 1R profit                            │
│    - Breakeven: SL → entry at 1.5R                          │
│    - Chandelier: ATR trail at 2R                             │
│    - Trailing SL: 1R→0.5R, 2R→1.5R, 4R→3R lock             │
│    - Queue Monitor: Liquidity Mirage detection              │
│                                                             │
│  CONFIDENCE DECAY:                                          │
│    - Only trails SL tighter (TPs preserved, NO cancel)      │
│                                                             │
│  BYBIT V5 API:                                              │
│    - PostOnly Maker entry + Stop-Market SL                  │
│    - Passive Maker close for time-stop                      │
└─────────────────────────────────────────────────────────────┘
```

### 1.2 ML Pipeline

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
ML Filters (8) → OMS Filters (15) → RiskManager → Order
```

### 1.3 Exit Pipeline

```
Position opened → Exit Grid Deployed (SL + TP)
     ↓
monitorExitOrders (500ms):
  - TP fills → breakeven SL → trailing SL
  - SL fills → finalize
  - 120s hard time-stop → passive Maker close
  - 180s breakeven → SL to fillPrice ± 0.13%
     ↓
PositionManager (500ms):
  - Scale-Out: 50% at 1R
  - Breakeven: SL → entry at 1.5R
  - Chandelier: ATR trail at 2R
  - Queue Monitor: wall evaporation → cancel
```

---

## 2. Конфигурация

| Параметр | Значение | Файл |
|----------|----------|------|
| PREDICT_PNL | true | docker-compose.yml |
| MIN_EDGE | 0.0005 | engine.py |
| TOXIC_THRESHOLD | 0.40 | engine.py |
| CONFIDENCE_THRESHOLD | 0.40 | .env |
| Effective Confidence Cap | 0.60 | service.go |
| Pattern Memory TTL | 1 hour | online_learner.py |
| PATTERN_SIMILARITY | 0.92 | .env |
| PATTERN_BLOCK_LIMIT | 50 | engine.py |
| Escalation threshold | 0.80 | engine.py |
| Hard Time-Stop | 120s | service.go |
| Breakeven | 180s | service.go |
| SL range | [0.3%, 0.8%] | exit_grid.go |
| R:R enforcement | ≥ 1.2 | exit_grid.go |
| Scale-out | 50% at 1R | position_manager.go |
| Max TP | 3% | exit_grid.go |
| RETRAIN_EPOCHS | 12 | docker-compose.yml |
| RETRAIN_TRADE_THRESHOLD | 10 | .env |

---

## 3. Статистика (backtest, 190 сделок)

| Конфигурация | PnL | EV/trade | R:R |
|-------------|-----|----------|-----|
| Old (SL 1.5%, R:R 0.7, 300s market) | -$9.87 | -$0.05 | 0.66 |
| SL 0.5%, R:R 1.2, 180s breakeven | -$3.66 | -$0.02 | 0.86 |
| SL 0.5%, R:R 1.2, 120s breakeven | **-$3.35** | **-$0.02** | **0.87** |

---

## 4. Идеи по улучшению

### 4.1 Решение корневой проблемы: R:R = 0.87 при WR = 49%

**Проблема**: Break-even WR = 50.7% при текущем R:R. WR = 49% — не хватает 1.7%.

**Идея 1: Увеличить MinTPPct с 0.003 до 0.005**
- TP станет дальше → R:R улучшится → breakeven WR снизится
- Риск: TP реже срабатывает (trade-off)
- Тест: прогон backtest с MinTPPct=0.005

**Идея 2: Ограничить Max SL по R:R**
- Текущий Max SL = 0.8%. Если TP < SL × 1.2 — reject сделку
- Это уже есть в коде (R:R enforcement ≥ 1.2), но TP может быть слишком далеко
- Тест: увеличить до R:R ≥ 1.5 (уже тестировали — хуже)

**Идея 3: Trailing TP вместо фиксированного**
- При profits > 0.2% начать двигать TP следом за ценой
- Ловит тренд лучше чем фиксированный TP
- Тест: добавить trailing TP на 50% от текущей прибыли

**Идея 4: Volume-weighted entry**
- Входить только при volume spike (volume > 2× SMA)
- Отсеивает входы в шум → улучшает WR
- Тест: добавить фильтр volume_spike в OMS

### 4.2 Улучшение предсказания направления

**Проблема**: LONG WR = 49%, SHORT WR = 49%. Модель не различает направления.

**Идея 1: Symbol-level features в state_vector**
- Добавить в 22 macro features: recent WR символа, loss count, avg PnL
- Модель увидит "этот символ плохо торгуется" и будет увереннее
- Тест: добавить 4 features → MACRO_DIM 22→26

**Идея 2: Per-symbol model**
- Обучить отдельную модель для топ-5 символов по volume
- Тест: BTC, ETH, SOL, XRP, DOGE — по отдельности

**Идея 3: Ensemble prediction**
- 3 модели: SHORT-prediction, LONG-prediction, HOLD-prediction
- Голосование → повышает точность направления
- Тест: 3 отдельных MLP, voting ensemble

### 4.3 Управление убытками

**Идея 1: Symbol-level hard block (как fallback)**
- Если символ имеет WR < 20% за 5+ сделок → заблокировать на 1 час
- Не вместо escalation, а как дополнительный слой

**Идея 2: Dynamic position sizing**
- Уменьшать размер позиции при losses streak
- Kelly: если 3+ losses подряд → снизить до 25% Kelly

**Идея 3: Correlation-based position limit**
- Если открытые позиции коррелируют > 0.8 → не открывать новые
- Предотвращает кластерные потери

### 4.4 Улучшение TP/SL

**Идея 1: ATR-based TP вместо fixed**
- TP = entry + ATR × multiplier
- Адаптируется к волатильности актива
- Тест: TP = entry + ATR × 1.5

**Идея 2: Liquidity-weighted SL**
- SL = уровень с наибольшей ликвидностью + буфер
- Уже реализовано (ComputeLiquiditySL), но можно усилить

**Идея 3: Multi-TP ladder**
- TP1: 50% позиции при 0.5% профите
- TP2: 50% позиции при ATR-based уровне
- Ловит и маленькие и большие движения

### 4.5 Backtest methodology

**Текущий подход**: реконструкция PnL из логов OMS + симуляция новых параметров.

**Улучшения**:
1. **InfluxDB replay**: загрузить реальные тиковые данные, прогнать через новый pipeline
2. **Walk-forward validation**: train на 6ч, test на 1ч, rolling
3. **Monte Carlo simulation**: 1000 итераций случайного порядка сделок
4. **Regime analysis**: отдельные метрики для Choppy/Trending/Breakout
5. **Symbol decomposition**: отдельные метрики для топ-10 символов

**Файлы для анализа**:
- `trade_stats_10h.json` — сырые данные 190 сделок
- `backtest_results.json` — результаты сравнения конфигураций
- `backtest_configs.py` — скрипт симуляции

**Промт для AI-анализа**: `ANALYSIS_PROMT.md` — полное описание системы для GPT/Claude
