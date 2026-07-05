# Анализ торговой системы: схема + промт для нейросети

---

## ЧАСТЬ 1: АРХИТЕКТУРА СИСТЕМЫ

### Общая схема

```
┌─────────────────────────────────────────────────────────────────────┐
│                        ДАННЫЕ (Bybit Demo API)                      │
│  Orderbook (bid/ask/size) + Trade flow (price/size/direction)       │
│  + Funding Rate + Price history                                     │
└──────────────────────────────┬──────────────────────────────────────┘
                               │ Redis Pub/Sub
                               ▼
┌──────────────────────────────────────────────────────────────────────┐
│                     ML ENGINE (Python 3.11 + ONNX GPU)              │
│                                                                      │
│  ┌──────────────┐  ┌──────────────┐  ┌─────────────────────────┐   │
│  │ OrderbookCNN │  │ FlowGRU      │  │ FeatureStore            │   │
│  │ (60×2→32)   │  │ +Attention   │  │ (22 macro features)     │   │
│  │              │  │ (60×3→32)    │  │                         │   │
│  └──────┬───────┘  └──────┬───────┘  └───────────┬─────────────┘   │
│         │                  │                      │                  │
│         └──────────────────┼──────────────────────┘                  │
│                            ▼                                         │
│                   ┌─────────────────┐                                │
│                   │ FusionModel     │                                │
│                   │ CNN(32)+GRU(32) │                                │
│                   │ +Macro(22→64)   │                                │
│                   │ = 128-dim state │                                │
│                   └────────┬────────┘                                │
│                            │                                         │
│                            ▼                                         │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │ DecisionMLP: state(128) + memory(8) → 136 → 6 outputs      │   │
│  │  [0:3] direction (LONG/SHORT/HOLD)                          │   │
│  │  [3]   confidence (0-1)                                     │   │
│  │  [4]   vol_mult (0.5-3.0)                                   │   │
│  │  [5]   trap_prob (0-1)                                      │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                      │
│  ┌─────────────────────────┐  ┌─────────────────────────────────┐  │
│  │ FAISS Vector Memory     │  │ Pattern Memory                  │  │
│  │ 1,753 winning patterns  │  │ 1,007 losing patterns           │  │
│  │ → 8-dim v_memory        │  │ → blocks signal if 3+ similar   │  │
│  │                         │  │   losses (cosine sim ≥ 0.92)    │  │
│  └─────────────────────────┘  └─────────────────────────────────┘  │
│                                                                      │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │ Online Learner (EWC)                                        │   │
│  │ - Updates DecisionMLP weights after every 5 trades          │   │
│  │ - Replay buffer (200 samples)                               │   │
│  │ - Loss penalty: 1.5x for losing trades                      │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                      │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │ Retrain Worker                                              │   │
│  │ - Triggers: every 10 winning trades OR every 2 hours        │   │
│  │ - Data: InfluxDB 48h lookback, 24 symbols                   │   │
│  │ - Epochs: 12, batch_size: 64, lr: 5e-4                      │   │
│  │ - Quality gate: acc_improved + no degradation                │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                      │
│  ФИЛЬТРЫ ДО ОТПРАВКИ В OMS (engine.py _run_tick_prediction):        │
│                                                                      │
│  1. Trap Head (trap_prob > 0.60 → vol_mult *= 0.5)                  │
│  2. Correlation filter (BTC corr > 0.85 → HOLD)                     │
│  3. Pattern Memory (3+ similar losses → block)                       │
│  4. Symbol+Setup escalation (3+ losses in setup → threshold 0.80)   │
│  5. Symbol cooldown (30-60min after loss)                            │
│  6. Dynamic confidence (WR < 40% → threshold raised)                 │
│  7. Trend filter (SHORT in uptrend → blocked/flipped to LONG)        │
│  8. Confidence threshold (direction-specific)                        │
│  9. Exit optimizer trade_score (score < 0.3 → block)                │
└──────────────────────────────┬──────────────────────────────────────┘
                               │ Redis Pub/Sub (signals channel)
                               ▼
┌──────────────────────────────────────────────────────────────────────┐
│                     OMS EXECUTION (Go)                               │
│                                                                      │
│  ФИЛЬТРЫ ДО ENTRY (service.go handleSignal → placeNewEntry):        │
│                                                                      │
│  1. Dynamic confidence per symbol (OMS-side tracking)                │
│  2. Spread filter (spread > 0.5% → reject)                          │
│  3. Zero depth filter (total depth = 0 → reject)                    │
│  4. OBI momentum (OBI > 0.2 against direction → reject)             │
│  5. Price trend (price moved > 0.5% against in 30s → reject)        │
│  6. Exchange position cross-check                                    │
│  7. RiskManager (ATR-based SL, EV check, Kelly sizing)              │
│                                                                      │
│  ENTRY:                                                              │
│  - Maker order (PostOnly) at best_bid/ask + entry_maker_ticks       │
│  - Peg reprice every tick until fill                                 │
│  - Timeout: 3600s                                                    │
│                                                                      │
│  POSITION MANAGEMENT:                                                │
│  - Exit grid: TP (liquidity > ml > fee_aware) + SL                  │
│  - TP max 3% ROI, Min TP dynamic (1% for <$0.01, 0.5% for <$0.10)  │
│  - SL via ComputeLiquiditySL (orderbook density zones)               │
│  - Adverse signal cooldown: 60s after TP redeploy                    │
│  - Confidence decay: trails SL (NO TP cancel)                        │
│  - Time-Stop: configurable per symbol                                │
│  - Scale-Out: partial profit taking                                  │
│  - Trailing stop: locks profit when in gain                          │
│  - Ghost position cleanup on startup (exchange cross-check)          │
└──────────────────────────────┬──────────────────────────────────────┘
                               │ Bybit Demo API (signed orders)
                               ▼
┌──────────────────────────────────────────────────────────────────────┐
│                     BYBIT DEMO EXCHANGE                              │
│  Deposit: $2,000  Leverage: 5x  Max concurrent: 200                 │
│  Fee: 0.055% maker / 0.075% taker                                   │
└──────────────────────────────────────────────────────────────────────┘
```

---

## ЧАСТЬ 2: ТЕКУЩИЕ МЕТРИКИ

### Баланс и PnL

| Метрика | Значение |
|---------|----------|
| Deposit | $2,000.00 |
| Balance | **$1,986.20** |
| PnL (24ч) | **-$13.80** |
| Trades | 224 (103W/121L) |
| Win Rate | **46%** |
| Avg win | **+$0.18** |
| Avg loss | **-$0.36** |
| Win/Loss ratio | **0.50** (нужен ≥1.0) |
| Break-even WR | **66%** |

### Поставщики PnL

| Причина | Кол-во | PnL | Avg |
|---------|:------:|-----|-----|
| take_profit | 100 | +$18.78 | +$0.19 |
| stop_loss | 81 | -$29.63 | -$0.37 |
| fee_loss | 6 | -$1.50 | -$0.25 |
| stale_tracker | 31 | $0.00 | $0.00 |
| exchange_closed | 6 | -$1.45 | -$0.24 |

### Win Rate по часам (UTC)

```
07:00  50%  -0.34
08:00  25%  -0.20
09:00  20%  +0.26
10:00  37%  -4.30   ← пик убытков
11:00  40%  -3.70
12:00  59%  +0.09
13:00  62%  +0.84   ← лучший час
14:00   0%  -0.96
15:00  54%  +0.02
16:00  47%  -0.22
17:00  48%  -1.39
18:00  18%  -2.91
19:00  40%  -0.99
```

### PnL по времени удержания

| Hold | Trades | PnL | Avg per trade |
|------|:------:|-----|:-------------:|
| <2мин | 46 | -$1.79 | -$0.04 |
| 2-10мин | 64 | -$2.53 | -$0.04 |
| >10мин | 114 | **-$9.48** | **-$0.08** |

### PnL по цене входа

| Entry Price | Trades | PnL |
|-------------|:------:|-----|
| <$0.01 | 42 | -$3.06 |
| $0.01-0.50 | 108 | **-$9.72** |
| >$0.50 | 74 | -$1.02 |

### LONG vs SHORT

| Dir | W | L | WR | PnL |
|-----|---|---|-----|-----|
| LONG | ~50 | ~60 | ~45% | - |
| SHORT | ~53 | ~61 | ~46% | - |

### Конфигурация

| Параметр | Значение |
|----------|----------|
| CONFIDENCE_THRESHOLD | 0.40 |
| MIN_SL_PCT | 0.006 (0.6%) |
| MIN_TP_PCT | 0.003 (0.3%) |
| MAX_TP_PCT | 0.02 (2%) |
| Leverage | 5x |
| Trade margin | $10 |
| PATTERN_SIMILARITY | 0.92 |
| PATTERN_BLOCK_LIMIT | 50 |
| RETRAIN_EPOCHS | 12 |
| MACRO_DIM | 22 |
| STATE_DIM | 128 |

---

## ЧАСТЬ 3: ПРОБЛЕМЫ

### Проблема 1: Win/Loss ratio = 0.50

TP срабатывает в 100 сделках, приносит +$18.78 (+$0.19/сделка).
SL срабатывает в 81 сделке, забирает -$29.63 (-$0.37/сделка).
**Средний TP в 2 раза меньше среднего SL.** R:R = 0.50.

### Проблема 2: >10мин удержание = -$9.48

114 сделок с hold >10мин генерируют **70% всех убытков**. Модель входит, TP не срабатывает, позиция держится долго и в итоге закрывается по SL.

### Проблема 3: Entry quality не фильтрует шум

46 сделок (<2мин) = immediate SL. Модель входит на пике, SL срабатывает сразу. Price trend filter работает, но не блокирует 15% таких сделок.

### Проблема 4: Microcap ($0.01-0.50) = -$9.72

108 сделок с entry $0.01-0.50 дают **70% убытков**. Для дешёвых монет SL в долларах больше чем TP (из-за tick size).

### Проблема 5: Model predicts equally badly

SHORT WR = 46%, LONG WR = 45%. Модель не различает направления рынка. В uptrend входит SHORT, в downtrend — LONG.

---

## ЧАСТЬ 4: ПРОМТ ДЛЯ НЕЙРОСЕТИ

Скопируйте этот промт в ChatGPT/Claude/Gemini для глубокого анализа:

---

```
Ты — эксперт по алгоритмической торговле на криптовалютных фьючерсах.

Перед тобой полное описание автоматической торговой системы. 
Проанализируй архитектуру, метрики и проблемы. Дай конкретные, 
практические рекомендации по улучшению прибыльности.

═══════════════════════════════════════════════════
АРХИТЕКТУРА СИСТЕМЫ
═══════════════════════════════════════════════════

Система торгует на Bybit USDT Perpetual Futures (Demo, $2,000, 5x leverage).

ДАННЫЕ:
- Orderbook (bid/ask/size) — обновляется в реальном времени
- Trade flow (price/size/direction) — агрессивные сделки
- Funding rate
- Price history (rolling)

ML PIPELINE:
1. OrderbookCNN: Conv1d на orderbook sequence (60 тиков × 2) → 32-dim embedding
2. FlowGRUAttention: GRU + self-attention на trade flow (60 × 3) → 32-dim embedding
3. Macro features: 22 признаков (obi, cvd, trend, funding, volume imbalance, liquidity depth)
4. Fusion: CNN(32) + GRU(32) + Macro(22→64) = 128-dim state vector
5. DecisionMLP: state(128) + memory(8) → direction[3] + confidence + vol_mult + trap_prob
6. FAISS memory: 1,753 winning trade patterns → 8-dim memory vector
7. Pattern Memory: 1,007 losing patterns → blocks if 3+ similar losses (cosine sim ≥ 0.92)

ФИЛЬТРЫ ДО ENTRY (9 штук):
1. Trap head blocks if trap_prob > 0.60
2. Correlation with BTC > 0.85 → HOLD
3. Pattern memory: 3+ similar losses → block
4. Symbol+Setup escalation: 3+ losses in same symbol/regime/direction → threshold 0.80
5. Symbol cooldown: 30-60 min after loss
6. Dynamic confidence: WR < 40% → threshold raised proportionally
7. Trend filter: SHORT blocked if trend_5m > 0; flipped to LONG if trend_5m > 0.3%
8. Confidence threshold: 0.40 base (LONG: 0.50)
9. Exit optimizer: trade_score < 0.3 → block

OMS (Go) — дополнительные фильтры:
1. Dynamic confidence (OMS-side tracking, cap 0.95)
2. Spread > 0.5% → reject
3. Zero depth → reject
4. OBI momentum ±0.2 against direction → reject
5. Price moved > 0.5% against in 30s → reject
6. RiskManager: ATR-based SL, EV check, Kelly sizing
7. PostOnly maker entry with peg reprice

EXIT MANAGEMENT:
- TP priority: liquidity_tp > ml_tp > fee_aware_tp
- Max TP 3% ROI, Min TP dynamic (1% for <$0.01 coins)
- SL via orderbook density zones (ComputeLiquiditySL)
- Confidence decay: trails SL only (does NOT cancel TPs)
- Adverse signal cooldown: 60s after TP redeploy
- Time-Stop: configurable per symbol

═══════════════════════════════════════════════════
МЕТРИКИ (24 часа)
═══════════════════════════════════════════════════

BALANCE: $2,000 → $1,986.20 (PnL = -$13.80)
TRADES: 224 total (103W / 121L, WR = 46%)

WIN/LOSS BREAKDOWN:
- take_profit: 100 trades, +$18.78 (avg +$0.19)
- stop_loss: 81 trades, -$29.63 (avg -$0.37)
- fee_loss: 6 trades, -$1.50
- stale_tracker: 31 trades, $0.00
- exchange_closed: 6 trades, -$1.45

HOLD TIME vs PnL:
- <2 min: 46 trades, -$1.79 (immediate SL)
- 2-10 min: 64 trades, -$2.53
- >10 min: 114 trades, -$9.48 (70% of total losses!)

ENTRY PRICE vs PnL:
- <$0.01: 42 trades, -$3.06
- $0.01-0.50: 108 trades, -$9.72 (70% of losses!)
- >$0.50: 74 trades, -$1.02

HOUR-BY-HOUR (UTC):
07: 50% WR, -0.34 | 08: 25%, -0.20 | 09: 20%, +0.26
10: 37%, -4.30 | 11: 40%, -3.70 | 12: 59%, +0.09
13: 62%, +0.84 | 14: 0%, -0.96 | 15: 54%, +0.02
16: 47%, -0.22 | 17: 48%, -1.39 | 18: 18%, -2.91
19: 40%, -0.99

KEY METRICS:
- Win/Loss ratio: 0.50 (loss is 2x win!)
- Avg win: +$0.18, Avg loss: -$0.36
- Break-even WR needed: 66% (only achieving 46%)
- LONG and SHORT both ~46% WR (model can't predict direction)

═══════════════════════════════════════════════════
ПРОШЛЫЕ ИСПРАВЛЕНИЯ (уже применены)
═══════════════════════════════════════════════════

1. Adverse signal cooldown: 60s after TP redeploy
2. TP drift threshold: 0.3% → 2%
3. Dynamic confidence per symbol (ML + OMS)
4. Entry quality filters (spread, OBI, price trend)
5. Confidence decay: no TP cancel
6. LONG trades enabled (not shadow-only)
7. Volume imbalance + Funding rate features
8. Retrain epochs 8→12, noise augmentation
9. Symbol+Setup escalation (0.80 threshold)
10. Pattern Memory symbol-level check
11. Cooldown 5min → 30-60min
12. PATTERN_BLOCK_LIMIT 10→50
13. Dynamic MinTPPct for cheap coins
14. Ghost position cleanup on startup

═══════════════════════════════════════════════════
ПРОМТ: ЧТО НУЖНО
═══════════════════════════════════════════════════

Проанализируй данные выше и ответь на вопросы:

1. ПРИЧИНА УБЫТОЧНОСТИ: почему R:R = 0.50? Что конкретно делает TP слишком маленьким 
   и SL слишком большим? Какие параметры нужно изменить?

2. ENTRY QUALITY: 46 сделок (<2мин) сразу в SL. 114 сделок (>10мин) = 70% убытков. 
   Как улучшить时机 входа? Нужны ли дополнительные фильтры?

3. MICROCAP PROBLEM: 108 сделок ($0.01-0.50) = -$9.72. Стоит ли торговать микрокапы 
   с текущим размером SL/TP? Или нужно изменить параметры для них?

4. DIRECTION PREDICTION: LONG WR = 45%, SHORT WR = 46%. Модель не различает 
   направления. Как улучшить предсказание направления?

5. TIME-OF-DAY: часы 10-11 и 18 дают -$12. Часы 12-13 дают +$0.93. 
   Стоит ли добавить time-of-day filter?

6. TP OPTIMIZATION: avg win = $0.19 vs avg loss = $0.37. Как увеличить 
   средний win? Trailing TP? Longer hold? Multi-TP ladder?

7. SL OPTIMIZATION: avg hold >10мин = -$0.08/trade. Это означает что SL 
   слишком далеко и не защищает. Как оптимизировать SL?

8. СИСТЕМНЫЕ ИЗМЕНЕНИЯ: какие изменения в архитектуре дадут максимальный 
   эффект? Ранжируй по ROI.

9. RISK MANAGEMENT: $2,000 deposit, 5x leverage, $10 per trade. 
   Оптимальный ли размер позиции? Нужен ли position sizing на основе Kelly?

10. RETRAIN: модель учится каждые 10 winning trades. Достаточно ли этого? 
    Нужно ли добавить данные в training?

Для каждого ответа:
- Конкретные числа и пороги
- Ожидаемый эффект на PnL
- Сложность внедрения (низкая/средняя/высокая)
- Приоритет (1-5, где 5 = критически важно)
```

---

## ЧАСТЬ 5: ДОПОЛНИТЕЛЬНЫЕ ДАННЫЕ ДЛЯ ПРОМТА

При необходимости добавьте в промт:

### Топ-5 убыточных сделок

| # | Entry | Exit | Hold | PnL | Reason |
|---|-------|------|------|-----|--------|
| 1 | 0.1090 | 0.1104 | 864s | -$0.70 | stop_loss |
| 2 | 0.0515 | 0.0521 | 2544s | -$0.63 | exchange_closed |
| 3 | 0.0284 | 0.0288 | 5435s | -$0.57 | stop_loss |
| 4 | 0.5379 | 0.5430 | 8s | -$0.54 | fee_loss |
| 5 | 0.0048 | 0.0047 | 6480s | -$0.54 | stop_loss |

### Топ-5 прибыльных сделок

| # | Entry | Exit | Hold | PnL | Reason |
|---|-------|------|------|-----|--------|
| 1 | — | — | — | +$0.62 | take_profit |
| 2 | — | — | — | +$0.56 | take_profit |
| 3 | — | — | — | +$0.52 | take_profit |
| 4 | — | — | — | +$0.49 | take_profit |
| 5 | — | — | — | +$0.48 | take_profit |

### Retrain

- Текущая модель: 20260705_051530
- Retrain cycles за 24ч: ~30
- FAISS: 1,753 winning entries
- Val accuracy: 0.61-0.65
- Precision/Recall: 68.9% (SHORT class)

### Pattern Memory

- Blocked signals за 24ч: ~170,000+
- Top blocked: SHORT in Choppy regime
- Similarity threshold: 0.92
- TTL: 0 (infinite)
