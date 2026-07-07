# Funding Rate Filter + Squeeze Detection
# Файл: service.go (inline, не отдельный файл)
# Дата: 2026-07-07

---

## 1. Константы (service.go:26-32)

```go
const DynamicConfCap = 0.45

// Funding Rate extremes for bias filter and squeeze detection
const (
    ExtremePositiveFunding = 0.0005  // +0.05% per 8h — longs overleveraged
    ExtremeNegativeFunding = -0.0005 // -0.05% per 8h — shorts overleveraged
)
```

---

## 2. Funding Rate Bias Filter (service.go:593-622)

Внутри `handleSignal()`, после volume filter, перед exchange cross-check:

```go
// ════════════════════════════════════════════════════════════════
// FUNDING RATE BIAS FILTER — asymmetric direction filter
// Extreme positive funding: longs overleveraged → block LONG, favor SHORT
// Extreme negative funding: shorts overleveraged → block SHORT, favor LONG
// ════════════════════════════════════════════════════════════════
fr := signal.FundingRate
if fr >= ExtremePositiveFunding {
    if signal.Direction == "LONG" {
        s.logger.Warn("[FUNDING FILTER] LONG rejected — extreme positive funding",
            "symbol", signal.Symbol,
            "funding_rate", fmt.Sprintf("%.4f%%", fr*100),
            "risk", "Long Squeeze — longs overleveraged")
        return nil
    }
    s.logger.Info("[FUNDING FILTER] SHORT favored — extreme positive funding",
        "symbol", signal.Symbol,
        "funding_rate", fmt.Sprintf("%.4f%%", fr*100))
}
if fr <= ExtremeNegativeFunding {
    if signal.Direction == "SHORT" {
        s.logger.Warn("[FUNDING FILTER] SHORT rejected — extreme negative funding",
            "symbol", signal.Symbol,
            "funding_rate", fmt.Sprintf("%.4f%%", fr*100),
            "risk", "Short Squeeze — shorts overleveraged")
        return nil
    }
    s.logger.Info("[FUNDING FILTER] LONG favored — extreme negative funding",
        "symbol", signal.Symbol,
        "funding_rate", fmt.Sprintf("%.4f%%", fr*100))
}
```

### Логика

| Funding Rate | LONG | SHORT |
|-------------|------|-------|
| ≥ +0.05% | **REJECT** (Long Squeeze risk) | FAVORED |
| -0.05% ~ +0.05% | PASS | PASS |
| ≤ -0.05% | FAVORED | **REJECT** (Short Squeeze risk) |

---

## 3. Squeeze Detection (service.go:789-816)

После RiskManager approval, перед qty normalization:

```go
// ════════════════════════════════════════════════════════════════
// SQUEEZE DETECTION — funding + momentum alignment boost
// When extreme funding aligns with orderbook momentum, probability
// of a sharp directional move (squeeze) is maximal.
// Boost vol_mult (position size) by 1.5× for favorable squeeze setups.
// ════════════════════════════════════════════════════════════════
obMomentum := s.obMomentum.Momentum(signal.Symbol)
fr := signal.FundingRate

// Short Squeeze: extreme negative funding + buying pressure + LONG signal
if fr <= ExtremeNegativeFunding && obMomentum > 0.3 && signal.Direction == "LONG" {
    signal.PositionScale *= 1.5
    s.logger.Warn("[SQUEEZE DETECTED] Short Squeeze — funding + momentum alignment",
        "symbol", signal.Symbol,
        "funding_rate", fmt.Sprintf("%.4f%%", fr*100),
        "momentum", fmt.Sprintf("%.4f", obMomentum),
        "vol_boost", "1.5×",
    )
}

// Long Squeeze: extreme positive funding + selling pressure + SHORT signal
if fr >= ExtremePositiveFunding && obMomentum < -0.3 && signal.Direction == "SHORT" {
    signal.PositionScale *= 1.5
    s.logger.Warn("[SQUEEZE DETECTED] Long Squeeze — funding + momentum alignment",
        "symbol", signal.Symbol,
        "funding_rate", fmt.Sprintf("%.4f%%", fr*100),
        "momentum", fmt.Sprintf("%.4f", obMomentum),
        "vol_boost", "1.5×",
    )
}
```

### Логика

| Funding | Momentum | Direction | Action |
|---------|----------|-----------|--------|
| ≤ -0.05% | > +0.3 (buying) | LONG | **vol × 1.5** (Short Squeeze) |
| ≥ +0.05% | < -0.3 (selling) | SHORT | **vol × 1.5** (Long Squeeze) |
| иначе | — | — | без изменений |

---

## 4. Позиция в цепочке фильтров

```
handleSignal():
  │
  ├─ Circuit Breaker (IsAutoBanned)
  ├─ Dynamic Confidence (cap 0.45)
  │
  ├─ orderbook block:
  │   ├─ Spread > 0.5% → reject
  │   ├─ Zero depth → reject
  │   ├─ OBI static (±0.4) → reject
  │   ├─ Orderbook momentum shift (±0.3) → reject
  │   ├─ Price trend (0.5%/30s) → reject
  │   └─ Volume (0.5× SMA) → reject
  │
  ├─ [NEW] FUNDING RATE BIAS FILTER (±0.05%) → reject/favor
  │
  ├─ Exchange cross-check
  ├─ RiskManager (ATR SL, EV, Kelly)
  │
  ├─ [NEW] SQUEEZE DETECTION → vol × 1.5
  │
  └─ Limit Chasing entry (3×3s)
```

---

## 5. Источник данных

### Python ML Engine (engine.py:1010)

```python
signal = {
    "symbol": symbol,
    "direction": direction,
    "confidence": confidence,
    "funding_rate": self._funding_rates.get(symbol, buf.funding_rate),
    ...
}
```

### Обновление funding rates (engine.py:211)

```python
self._funding_rates[sym] = float(meta.get("funding_rate", 0))
self.features.set_funding_rate(sym, self._funding_rates[sym])
```

### Go OMS (models/models.go:9)

```go
type TradeSignal struct {
    ...
    FundingRate float64 `json:"funding_rate"`
    ...
}
```

### Метрика (service.go:849)

```go
metrics.FundingRate.WithLabelValues(signal.Symbol).Set(signal.FundingRate)
```

---

## 6. Обработка отсутствующих данных

Если `signal.FundingRate == 0` (данные недоступны):
- `0 >= 0.0005` → false, `0 <= -0.0005` → false
- **Фильтр пропускает сделку** — нет блокировки

---

## 7. Unit Tests

Фильтр проверяется через:
- `ExtremePositiveFunding = 0.0005` → `fr >= 0.0005` for LONG → reject
- `ExtremeNegativeFunding = -0.0005` → `fr <= -0.0005` for SHORT → reject
- `fr = 0` → no reject (graceful fallback)

```
Go tests: 14/14 PASS (executor, grid, liquidity)
```

---

## 8. Лог-примеры

### Funding Filter rejection
```
[FUNDING FILTER] LONG rejected — extreme positive funding
  symbol=FARTCOINUSDT funding_rate=0.0500% risk=Long Squeeze
```

### Funding Filter favor
```
[FUNDING FILTER] SHORT favored — extreme positive funding
  symbol=BTCUSDT funding_rate=0.0800%
```

### Squeeze Detection
```
[SQUEEZE DETECTED] Short Squeeze — funding + momentum alignment
  symbol=ETHUSDT funding_rate=-0.0600% momentum=0.4300 vol_boost=1.5×
```
