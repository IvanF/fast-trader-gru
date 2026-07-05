# Анализ входных данных OrderbookCNN: глубина стакана и нормализация

## Текущая архитектура

### Глубина стакана

**Top 10 уровней** bid + ask агрегируются в одно число в `features.py:65-66`:

```python
def add_orderbook(self, ts_ms: int, bids: list, asks: list) -> None:
    bid_vol = sum(_level_size(b) for b in bids[:10])  # top 10 bid
    ask_vol = sum(_level_size(a) for a in asks[:10])  # top 10 ask
```

### Формирование входного тензора

```
orderbook_sequence() → tensor (60, 2):
  channel[0] = obi = (bid_vol - ask_vol) / (bid_vol + ask_vol)  → [-1, +1]
  channel[1] = bid_vol - ask_vol                                  → raw разница
```

**Длина последовательности**: 60 тиков (обновлений orderbook)  
**Размер входа CNN**: `(batch, 2, 60)` — OB_DIM=2 канала × SEQ_LEN=60

### Архитектура CNN

```python
OrderbookCNN:
  Conv1d(2 → 16, kernel=3, padding=1) → ReLU
  Conv1d(16 → 32, kernel=3, padding=1) → ReLU
  AdaptiveAvgPool1d(1)
  FC(32 → 32)
  output: (batch, 32)
```

## Где берутся данные

| Источник | Описание |
|----------|----------|
| `bids[:10]` | Top 10 bid уровней orderbook |
| `asks[:10]` | Top 10 ask уровней orderbook |
| `add_orderbook()` | Вызывается при каждом обновлении orderbook |
| `points` deque | Rolling window 300 секунд (5 минут) |

## Проблемы

### Проблема 1: Channel 1 не нормализован

```
Channel 0: obi = (bid - ask) / (bid + ask)  → [-1, +1] ✅ нормализован
Channel 1: bid - ask                         → raw разница, разный масштаб ❌
```

**Пример**: BTCUSDT bid-ask = ±100 BTC, PENGUUSDT bid-ask = ±10,000,000 токенов. Разница в 100,000x. CNN учится на доминирующем масштабе BTC → сигналы от дешёвых монет тонут в шуме.

### Проблема 2: Top 10 агрегируются в одно число

```
Level 1: 50 BTC  ─┐
Level 2: 30 BTC   │
Level 3: 20 BTC   ├── sum = 100 BTC (одно число)
Level 4:  0 BTC   │
...               │
Level 10: 0 BTC  ─┘
```

CNN не видит:
- Есть ли **стенка** (wall) на 5-м уровне или стакан пустой
- **Концентрацию** liquidity (top-3 vs top-10)
- **Профиль глубины** (распределение объёмов по уровням)

### Проблема 3: Временна́я дискретизация

Orderbook обновляется при каждом изменении стакана. Частота зависит от актива:
- BTCUSDT: ~10-50 обновлений/сек → 60 тиков ≈ 1-6 секунд
- Микрокапы: ~1-5 обновлений/сек → 60 тиков ≈ 12-60 секунд

**Разные активы имеют разное временна́е разрешение** в одном и том же тензоре.

### Проблема 4: liquidity_features() отдельно от CNN

```python
# liquidity_features() — 7 признаков из top 10/3 уровней:
depth_imbalance, depth_concentration, spread_bps,
fill_to_depth, level_density, tanh(bid_vol), tanh(ask_vol)

# Идут в macro projection → FC → 64-dim state vector
# НЕ в CNN
```

Информация о глубине дублируется: частично в CNN (агрегированная), частично в macro (детальная). Но CNN не получает структурированные данные о глубине.

## Сравнение: что теряется vs что даётся CNN

| Параметр | Текущее | Оптимальное |
|----------|---------|-------------|
| Глубина | Top 10 агрегировано | Top 20-50 отдельных уровней |
| Нормализация | Channel 0 ✅, Channel 1 ❌ | Все каналы в [-1, +1] |
| Профиль глубины | Нет | Да (volume profile) |
| Wall detection | Нет | Да (unusual level sizes) |
| Spread | Нет в CNN | Да (best_ask - best_bid) |
| Частота тиков | Разная для разных символов | Фиксированная (1s бакеты) |

## Рекомендации

### 1. Нормализовать Channel 1

```python
# Было:
seq.append([obi, p.bid_vol - p.ask_vol])

# Стало:
raw_diff = p.bid_vol - p.ask_vol
normalized_diff = np.tanh(raw_diff / 1e6)  # или /avg_volume
seq.append([obi, normalized_diff])
```

### 2. Увеличить глубину и добавить профиль

```python
# Вход CNN: (batch, 6, 60) вместо (batch, 2, 60)
# Channel 0: obi (top-10 normalized)
# Channel 1: normalized volume difference
# Channel 2: spread_bps (best_ask - best_bid)
# Channel 3: depth_concentration (top-3 / top-10)
# Channel 4: wall_flag (1 если max_level > 3×median)
# Channel 5: level_density (count non-zero levels)
```

### 3. Стандартизировать временна́е разрешение

```python
# Вместо raw ticks: бакеты по 1 second
# Каждый бакет: avg OBI за эту секунду
# Или: time-weighted average
```

### 4. Добавить spread в CNN

Spread — критический сигнал. Сейчас spread считается в liquidity_features() отдельно. Добавить как канал в CNN даст модели возможность корреляционно анализировать spread + depth.

### 5. Wall detection как отдельный канал

```python
def wall_flag(self) -> float:
    """1.0 если самый большой уровень > 3× медианы."""
    sizes = [_level_size(b) for b in self.latest_bids[:10]]
    if not sizes:
        return 0.0
    median = np.median(sizes)
    max_size = max(sizes)
    return 1.0 if max_size > median * 3 and median > 0 else 0.0
```

## Файлы для изменения

| Файл | Что менять |
|------|-----------|
| `ml_engine/src/features.py` | `add_orderbook()`: хранить все уровни, не только sum. `orderbook_sequence()`: добавить каналы |
| `ml_engine/src/models/nn_models.py` | `OrderbookCNN`: `OB_DIM=2→6`, обновить FC |
| `ml_engine/src/influx_join.py` | `_rows_to_sequences()`: восстанавливать новые каналы из InfluxDB |
| `ml_engine/scripts/train.py` | Обновить `OB_DIM` константу |

## Верификация

1. Проверить что новый CNN совместим со старыми ONNX моделями (ONNX inference)
2. Проверить что InfluxDB хранит достаточно данных для восстановления новых каналов
3. Обучить модель с новыми каналами и сравнить val_acc со старой
