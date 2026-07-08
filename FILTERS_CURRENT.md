# Fast Trader GRU — Список текущих фильтров

_Обновлено: 2026-07-08 17:50 UTC_

---

## 1. ML Pipeline (engine.py)

### Основные пороги

| Параметр | Значение | Строка | Описание |
|----------|----------|--------|----------|
| PREDICT_PNL | true | env | PnL regression mode |
| Confidence cutoff | 0.0035 | L965 | min |pred_pnl| для ненулевого confidence |
| MIN_EDGE | 0.0035 | L1198 | Минимальный expected PnL (0.35%) |
| TRAP_THRESHOLD | 0.60 | L1177 | Порог trap detection |
| TOXIC_THRESHOLD | 0.40 | L1191 | Порог toxic flow block |
| CORRELATION_BLOCK_LIMIT | 8 | L1204 | Макс. коррелированных блоков за цикл |
| PATTERN_BLOCK_LIMIT | 50 | L1219 | Макс. pattern блоков за цикл |
| EVENT_DRIVEN_CONF | 0.85 | L1452 | Min conf для Event-Driven Gate |
| OBI_IMPULSE_THRESHOLD | 0.4 | L1453 | Min OBI impulse для Event-Driven Gate |

### KNIFE-GUARD (защита LONG)

| Параметр | Значение | Строка | Описание |
|----------|----------|--------|----------|
| BTC velocity threshold | -0.0002 | L1153 | Soft bearish drift detection |
| BTC correlation gate | > 0.60 | L1161 | corr с BTC для блокировки |
| OBI LONG block | < -0.1 | L1170 | Аски доминируют → блок LONG |
| GK LONG threshold | 0.65 | L1482 | Консервативный порог Gatekeeper |
| GK SHORT threshold | 0.45 | L1484 | Пермиссивный порог Gatekeeper |

---

## 2. OMS Entry Filters (service.go)

| Параметр | Значение | Описание |
|----------|----------|----------|
| Spread filter | > 0.5% | Отклонение при широком спреде |
| Volume LONG | ≥ 1.2×SMA(20) | Активное давление покупателей |
| Volume SHORT | ≥ 0.5×SMA(20) | Мёртвый рынок |
| Correlation block | > 0.70 | BTC corr > 0.70 + conf < 0.95 |
| LONG cluster max | 2 | Макс. коррелированных LONG позиций |
| Dynamic confidence | WR < 40% → raise | Штраф за низкий WR |
| Setup escalation | 3+ losses → 0.80 | Повышенный порог для токсичных setups |
| Trend filter | downtrend → conf=0.30 | LONG в нисходящем тренде |
| Funding rate block | ±0.05% | Экстремальный funding |

---

## 3. Pattern Memory (online_learner.py)

| Параметр | Значение | Строка | Описание |
|----------|----------|--------|----------|
| PATTERN_SIMILARITY_THRESHOLD | 0.85 | L32 | Cosine similarity порог |
| PATTERN_TTL_HOURS | 0.5 | L33 | TTL = 30 минут |
| CONSECUTIVE_LOSS_THRESHOLD | 10 | L34 | Порог последовательных проигрышей |
| Symbol-level window | TTL × 3600 | L124 | Окно для symbol-level block |
| Min losses to block | 5 | L97 | Мин. кол-во похожих паттернов |
| Min avg loss | -$0.10 | L99 | Мин. средний убыток для блокировки |

---

## 4. Gatekeeper (gatekeeper.py)

| Параметр | Значение | Строка | Описание |
|----------|----------|--------|----------|
| GK_THRESHOLD (fallback) | 0.55 | L23 | Порог если env не задан |
| GK_MIN_SAMPLES | 50 | L24 | Мин. сэмплов для обучения |
| GK_RETRAIN_EVERY | 50 | L25 | Интервал переобучения |
| Actual LONG threshold | 0.65 | engine.py L1482 | Консервативный (LONG) |
| Actual SHORT threshold | 0.45 | engine.py L1484 | Пермиссивный (SHORT) |

### Модель

| Метрика | Значение |
|---------|----------|
| Backend | CatBoost |
| AUC | 0.7842 |
| Accuracy | 79.17% |
| Training samples | 641 |
| Top features | atr_pct(13.4), open_positions(11.3), confidence(9.9) |

---

## 5. Time-Stop FSM (service.go)

| Параметр | Значение | Описание |
|----------|----------|----------|
| Normal time-stop | 180s | Unified для всех confidence |
| HFT time-stop | 60s | Scalping mode |
| Passive window | 5s | PostOnly → Kill-Switch timeout |
| Zombie retry | 60s | hardTimeStop + 60s → force close |
| Breakeven timer | 90s | SL → fillPrice + fees |
| Stale guard grace | 60s | Не ghost-check до 60s hold |
| Kill-Switch prefix | [KILL-SWITCH] | Аудит логов |

---

## 6. Exit Management (density_exit_manager.go)

| Параметр | Значение | Описание |
|----------|----------|----------|
| Wall ratio threshold | 15.0 | bid/ask depth ratio |
| Velocity threshold | 0.4 | OBI momentum reversal |
| Stagnation time | 90s | Hold time для stagnation |
| Stagnation R | 0.15 | CurrentR threshold |
| TP push pct | 0.2% | Сдвиг TP при wall detection |

---

## 7. Risk Management

| Параметр | Значение | Описание |
|----------|----------|----------|
| R:R target | 1.2 | Risk:Reward ratio |
| Min SL | 0.6% | Минимальный stop loss |
| Max SL | 0.8% | Максимальный stop loss |
| Min TP | 0.3% | Минимальный take profit |
| Max TP | 2.0% | Максимальный take profit |
| Fee breakeven | 0.15% | taker(0.075%) + maker(0.055%) |
| Leverage | 5× | Bybit Demo |
| Margin per trade | $10 | Notional = $50 |

---

## 8. Smart Labeling (MFE)

| Параметр | Значение | Описание |
|----------|----------|----------|
| MFE threshold | 75% | Если MFE > 75% TP distance → label=1 |

---

## 9. docker-compose Environment Variables

| Переменная | Значение | Сервис |
|-----------|----------|--------|
| CONFIDENCE_THRESHOLD | 0.85 | ml_engine |
| LONG_CONFIDENCE_THRESHOLD | 0.50 | ml_engine |
| KILL_CONFIDENCE_THRESHOLD | 0.35 | ml_engine |
| MAX_SIGNAL_CONFIDENCE | 0.98 | ml_engine |
| MIN_PNL_THRESHOLD | 0.0035 | ml_engine |
| PREDICT_PNL | true | ml_engine |
| GATEKEEPER_THRESHOLD | 0.55 | ml_engine (override в коде: LONG=0.65, SHORT=0.45) |
| TIME_STOP_SECONDS | 180 | oms_execution |
| HFT_TIME_STOP_SEC | 60 | oms_execution |
| HFT_BREAKEVEN_SEC | 30 | oms_execution |
| PATTERN_TTL_HOURS | 0.5 | online_learner (env override) |
| PATTERN_SIMILARITY_THRESHOLD | 0.85 | online_learner (env override) |
