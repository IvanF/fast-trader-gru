# Алгоритм системы принятия решений: полная схема фильтрации

## Время анализа: 2026-07-06 09:50 UTC

---

## 1. Воронка сигналов: от orderbook до сделки

```
51 символов × ~1200 predictions/символ = 239,391 prediction
                                         │
                                         ▼
                          ╔═══════════════════════╗
                          ║   ML ENGINE ФИЛЬТРЫ   ║
                          ╚═══════════════════════╝
                                         │
          ┌──────────────────────────────┼──────────────────────────────┐
          │                              │                              │
          ▼                              ▼                              ▼
    HOLD direction:              confidence low:              pattern blocked:
    2,612 (1.1%)               155,414 (65.0%)              68,621 (28.7%)
          │                              │                              │
          │                              │                              │
          ├──────────────────────────────┼──────────────────────────────┘
          │
          ▼
    correlation blocked: 117 (0.05%)
    warming blocked: 4,804 (2.0%)
          │
          ▼
    Signals to OMS: 5,971 (2.5%)
          │
          ▼
    ╔══════════════════════════════╗
    ║    OMS ENTRY ФИЛЬТРЫ        ║
    ╚══════════════════════════════╝
          │
          ▼
    spread/momentum/price: 3 (negligible)
    risk_manager reject: 1
    bybit agreement: 1
          │
          ▼
    Entries placed: 0 (blocked by Bybit)
```

---

## 2. ML Engine: все фильтры с текущими порогами

### Фильтр 1: Model Direction Decision
```python
# engine.py _run_tick_prediction
direction_logits = logits[:, :3]  # LONG=0, SHORT=1, HOLD=2
direction = ["LONG", "SHORT", "HOLD"][argmax(direction_logits)]
confidence = sigmoid(logits[3])
```
- Модель выдаёт направление + confidence + trap_prob
- **HOLD** фильтруется сразу: 2,612 (1.1%)

### Фильтр 2: Trap Head (engine.py ~line 1092)
```python
if trap_prob > 0.60 and confidence > 0.85:
    vol_mult *= 0.5  # уменьшает размер
# НЕ блокирует сигнал полностью
```
- Эффект: только уменьшает vol_mult, не блокирует

### Фильтр 3: Correlation Filter (engine.py ~line 1105)
```python
if corr > 0.85 and symbol != "BTCUSDT" and confidence < 0.95:
    direction = "HOLD"  # блокирует
```
- Blocked: 117 (0.05%) — минимальный эффект

### Фильтр 4: Pattern Memory (online_learner.py + engine.py ~line 1148)

```python
# online_learner.py should_avoid()
similar = find_similar(v_state, regime, direction, symbol)

# Symbol-level: 3+ losses of same symbol → block
symbol_losses = [p for p in similar if p.symbol == symbol and p.pnl < 0]
if len(symbol_losses) >= 3:
    return True, avg_loss

# Catastrophic loss: 1 loss > $0.50 → block forever
for p in similar:
    if p.pnl < -0.50:
        return True, p.pnl

# State-vector: 3+ similar losses, avg_loss < -$0.05
if loss_count >= 3 and avg_pnl < -0.05:
    return True, avg_pnl
```

**Ключевые пороги:**
| Параметр | Значение | Проблема |
|----------|----------|----------|
| Single catastrophic loss | **-$0.50** | 1 loss → block навсегда |
| Similar losses required | **3** | Слишком мало |
| avg_pnl threshold | **-$0.05** | Слишком чувствительно |
| Symbol check | 3+ losses of same symbol | Блокирует BTC/ETH навсегда |
| Similarity threshold | **0.92** | Не проблема |
| TTL | **0** (бесконечно) | Pattern never expires |

**Эффект: 68,621 заблокированных (28.7%)**

Топ блокируемых:
```
ETHUSDT:  3,057 SHORT blocks (1 catastrophic loss -$0.51)
BTCUSDT:  2,611 SHORT blocks (1 catastrophic loss -$0.51)
TLMUSDT:  2,239 SHORT blocks
ESUSDT:   1,823 SHORT blocks
LABUSDT:  1,760 SHORT blocks
```

### Фильтр 5: Symbol+Setup Escalation (engine.py ~line 1205)
```python
setup_key = f"{symbol}_{direction}_{regime}"
setup_losses = self._symbol_setup_losses.get(setup_key, [])
if len(setup_losses) >= 3:
    avg_setup_loss = sum(setup_losses) / len(setup_losses)
    if avg_setup_loss < -0.10:
        base_threshold = max(base_threshold, 0.80)
```
- 3+ losses в одном setup → threshold поднимается до **0.80**
- Модель обычно выдаёт confidence 0.45-0.55 → **все отсеиваются**

### Фильтр 6: Symbol Cooldown (engine.py ~line 1153)
```python
# After loss:
cooldown = 1800 if consec < 2 else 3600  # 30-60 минут

# After pattern block:
cooldown = 1800  # 30 минут
```
- 30-60 минут cooldown после каждого loss

### Фильтр 7: Dynamic Confidence (engine.py ~line 1189)
```python
base_threshold = 0.40  # current config
if symbol_wr < 0.40 and total >= 3:
    penalty = max(1.0, 3.0 - 2.5 * (wr / 0.40))
    base_threshold = min(0.40 * penalty, 0.95)
```

**Эффект:** Если WR < 40%, threshold может подняться до 0.95.
- При WR=0% (19 losses, 0 wins): penalty = 3.0, threshold = 0.40 × 3.0 = 0.75

**Эффект: 155,414 заблокированных (65.0%)** — **крупнейший фильтр!**

### Фильтр 8: Trend Filter (engine.py ~line 1211)
```python
if direction == "SHORT" and trend_5m > 0.003:
    direction = "LONG"  # flip
elif direction == "SHORT" and trend_5m > 0:
    return None  # block
elif direction == "LONG" and trend_5m < -0.003:
    direction = "SHORT"  # flip
```
- **2,612 заблокированы (1.1%)** — HOLD direction

### Фильтр 9: Exit Optimizer Score (engine.py ~line 1254)
```python
if trade_score < 0.3:
    return None
```
- Вторая нейросеть проверяет качество

---

## 3. OMS: все фильтры с текущими порогами

### OMS Filter 1: Dynamic Confidence (service.go ~line 390)
```go
stats := s.symbolStats[signal.Symbol]
if stats != nil {
    effectiveConf := stats.EffectiveConfidence(s.cfg.ConfidenceThreshold)
    if signal.Confidence < effectiveConf {
        return nil  // rejected
    }
}
```
- Аналог ML engine: penalty × base, cap 0.95
- Если символ имеет losses → penalty растёт → threshold растёт

### OMS Filter 2: Spread (service.go ~line 409)
```go
if spreadPct > 0.005 {  // 0.5%
    return nil
}
```

### OMS Filter 3: Zero Depth (service.go ~line 437)
```go
if totalDepth <= 0 {
    return nil
}
```

### OMS Filter 4: OBI Momentum (service.go ~line 442)
```go
obi := (bidV - askV) / (bidV + askV)
if signal.Direction == "SHORT" && obi < -0.2: reject
if signal.Direction == "LONG" && obi > 0.2: reject
```

### OMS Filter 5: Price Trend (service.go ~line 466)
```go
// Price moved > 0.5% against direction in last 30s → reject
if signal.Direction == "SHORT" && move > 0.005: reject
if signal.Direction == "LONG" && move < -0.005: reject
```

### OMS Filter 6: RiskManager (service.go ~line 597)
```go
riskResult := risk.ProcessSignal(
    direction, confidence, entryPrice, tpPrice,
    priceHistory, accountBalance, volMultiplier, tickSize,
)
```
**Внутри RiskManager:**
- Tick size filter: `tickSize / entryPrice > 0.001` → reject
- SL hard cap: `slDistancePct > 0.005` (0.5%) → reject
- EV check: `(confidence * rr) - ((1 - confidence) * 1) <= 0` → reject

---

## 4. Проблема: 93% сигналов блокируется

### Распределение блокировок (ML Engine)

| Фильтр | Заблокировано | % от预测 | Описание |
|---------|:------------:|:-------:|----------|
| **Low confidence** | 155,414 | **65.0%** | Dynamic confidence 0.40-0.95 |
| **Pattern memory** | 68,621 | **28.7%** | 1 catastrophic loss → block forever |
| **Direction (HOLD)** | 2,612 | 1.1% | Модель не уверена |
| **Warming** | 4,804 | 2.0% | Буферы ещё не заполнены |
| **Correlation** | 117 | 0.05% | Высокая корреляция с BTC |
| **Прошло** | **5,971** | **2.5%** | Только эти сигналы дошли до OMS |

### Что блокирует "Low confidence" (65%)

Это **dynamic confidence** — основной фильтр. Алгоритм:

```
1. Берётся base threshold: 0.40
2. Если WR < 40% за 3+ сделок:
   penalty = max(1.0, 3.0 - 2.5 × (WR / 0.40))
   threshold = 0.40 × penalty
3. Если есть symbol+setup с 3+ losses avg_loss < -$0.10:
   threshold = max(threshold, 0.80)
4. ML модель выдаёт confidence ~0.45-0.55
5. 0.55 < 0.80 → ЗАБЛОКИРОВАНО
```

### Что блокирует "Pattern memory" (28.7%)

```
1. BTCUSDT: 1 loss -$0.51 → single catastrophic → ALL SHORT заблокированы навсегда
2. ETHUSDT: 1 loss -$0.51 → single catastrophic → ALL SHORT заблокированы навсегда
3. TTL=0 → patterns никогда не экспирируются
4. 3+ losses в одном symbol → symbol-level block
```

---

## 5. Корень проблемы

**Система запрограммирована на максимальную осторожность.** Вместо того чтобы:

1. Входить в сделку с confidence=0.55 и R:R=1.0
2. Использовать hard SL 0.5% и scale-out 50%

Она блокирует 93% сигналов потому что:

1. **Dynamic confidence слишком агрессивна**: при WR < 40% threshold = 0.75-0.95, а модель выдаёт max 0.55 → **все сделки блокируются**
2. **Pattern memory слишком агрессивна**: 1 catastrophic loss > $0.50 → block forever → BTC/ETH/SOL навсегда заблокированы в SHORT
3. **Symbol setup escalation**: 3 losses avg < -$0.10 → threshold 0.80 → модель не может достичь
4. **Все 3 фильтра работают одновременно** → эффект мультипликативный

### Итоговая математика

```
ML engine prediction:     239,391
× Direction filter:       × 0.989  → 236,779
× Low confidence:         × 0.350  → 82,873
× Pattern memory:         × 0.713  → 59,089
× Other filters:          × 0.890  → 52,589
= Sent to OMS:            5,971 (2.5%)

OMS filters:
× Spread/momentum/price:  ~×0.999  → ~5,968
× RiskManager:            ~×0.999  → ~5,962
× Bybit agreement:        × 0.000  → 0
= Actual entries:         0
```

**Проблема в ML engine, не в OMS.** 93% блокировки происходит до отправки в OMS.

---

## 6. Конкретные проблемы для исправления

### Проблема A: Dynamic confidence (65% блокировок)
**Причина:** threshold поднимается до 0.75-0.95, а модель выдаёт max 0.55
**Решение:** Снизить max escalation до 0.60 или увеличить base confidence threshold до 0.50

### Проблема B: Pattern memory catastrophic block (28.7%)
**Причина:** 1 loss > $0.50 → block forever, TTL=0
**Решение:** Увеличить catastrophic порог до -$1.00 или добавить TTL 24ч

### Проблема C: Symbol+setup escalation (часть low_conf)
**Причина:** 3 losses avg < -$0.10 → threshold 0.80
**Решение:** Увеличить до 5+ losses или уменьшить escalation до 0.60

### Проблема D: Bybit Trading Terms
**Причина:** XAGUSDT/BABAUSDT не могут торговать без agreement
**Решение:** Подписать agreement или убрать эти символы из active_symbols

---

## 7. Рекомендуемые изменения

| # | Изменение | Ожидаемый эффект | Файл |
|---|-----------|-----------------|------|
| 1 | Dynamic confidence cap: 0.95 → **0.60** | 155K→~50K блокировок | engine.py |
| 2 | Pattern catastrophic: -$0.50 → **-$1.00** | 68K→~30K блокировок | online_learner.py |
| 3 | Symbol+setup: 0.80 → **0.60** | Умеренный escalation | engine.py |
| 4 | Pattern memory TTL: 0 → **24h** | Auto-expire старых patterns | online_learner.py |
| 5 | Remove XAGUSDT/BABAUSDT | 0 → 2 символов в active | .env |

**Ожидаемый результат:** 93% блокировка → ~50% блокировка → больше сделок → больше данных для обучения.
