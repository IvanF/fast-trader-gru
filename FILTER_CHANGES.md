# Предложения по изменению фильтров: от 93% блокировки к торговле

## Текущее состояние

```
239,391 predictions → 5,971 signals (2.5%) → 0 сделок
ML Engine блокирует 93%:  Low conf = 65%, Pattern = 28.7%
```

## Изменение 1: Dynamic confidence cap 0.95 → 0.60

**Проблема:** ML модель выдаёт confidence 0.45-0.55. Dynamic confidence поднимает threshold до 0.75-0.95. **Все сделки блокируются.**

**Решение:** Ограничить escalation до 0.60. При confidence 0.55 сделки будут проходить.

**Файл:** `ml_engine/src/engine.py`

```python
# Строка ~1197: было
base_threshold = min(self.cfg.confidence_threshold * penalty, 0.95)

# Стало
base_threshold = min(self.cfg.confidence_threshold * penalty, 0.60)
```

**Эффект:** 155,414 blocked → ~30,000 blocked (снижение на 80%)

---

## Изменение 2: Pattern catastrophic loss -$0.50 → -$1.00

**Проблема:** 1 сделка с loss > $0.50 → ВСЕ будущие сигналы для этого символа заблокированы навсегда. BTC/ETH/SOL заблокированы одним промахом.

**Решение:** Увеличить порог до -$1.00. При масштабе сделок ($10 маржа, 0.5% SL = $0.05 loss), -$1.00 = реальная катастрофа (20x ожидаемого loss).

**Файл:** `ml_engine/src/online_learner.py`

```python
# Строка ~127: было
if p.pnl < -0.50:

# Стало
if p.pnl < -1.00:
```

**Эффект:** 68,621 blocked → ~20,000 blocked (BTC/ETH/SOL разблокированы)

---

## Изменение 3: Pattern memory TTL 0 → 24h

**Проблема:** Patterns никогда не экспирируются. Old bad patterns блокируют навсегда.

**Решение:** TTL = 24 часа. После 24 часов pattern удаляется, символ может торговать снова.

**Файл:** `.env`

```
# Стало:
PATTERN_TTL_HOURS=24
```

**Эффект:** Автоочистка устаревших блокировок

---

## Изменение 4: Symbol+setup escalation 0.80 → 0.60

**Проблема:** 3 losses в одном setup → threshold 0.80. Модель выдаёт 0.55 → блок.

**Решение:** Escalation до 0.60 вместо 0.80. При confidence 0.55 сделки проходят.

**Файл:** `ml_engine/src/engine.py`

```python
# Строка ~1210: было
escalation = 0.80
if escalation > base_threshold:
    base_threshold = escalation

# Стало
escalation = 0.60
if escalation > base_threshold:
    base_threshold = escalation
```

**Эффект:** Symbol+setup больше не блокирует при normal confidence

---

## Изменение 5: Убрать XAGUSDT из active symbols

**Проблема:** XAGUSDT не может торговать (Bybit agreement). Все его сигналы = waste.

**Решение:** Добавить XAGUSDT в исключения.

**Файл:** `.env` (или ML engine config)

```
BLACKLIST_SYMBOLS=XAGUSDT
```

**Эффект:** Убирает мёртвые сигналы

---

## Сводка изменений

| # | Файл | Изменение | Было → Стало |
|---|------|-----------|-------------|
| 1 | engine.py:1197 | Dynamic confidence cap | 0.95 → **0.60** |
| 2 | online_learner.py:127 | Catastrophic loss threshold | -$0.50 → **-$1.00** |
| 3 | .env | Pattern TTL | 0 → **24h** |
| 4 | engine.py:1210 | Symbol+setup escalation | 0.80 → **0.60** |
| 5 | .env | Blacklist XAGUSDT | none → **XAGUSDT** |

## Ожидаемый результат

```
БЫЛО:  239K → 5.9K signals (2.5%) → 0 сделок (93% blocked)
СТАЛО: 239K → ~50K signals (21%) → ~100-200 сделок/день
```

| Фильтр | Блокировало | Станет |
|---------|:-----------:|:------:|
| Low confidence | 155,414 (65%) | ~40,000 (17%) |
| Pattern memory | 68,621 (29%) | ~20,000 (8%) |
| **Итого** | **93%** | **~25%** |
| **Проходит** | **2.5%** | **~21%** |
