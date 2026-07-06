# Plan: Multimodal Early Fusion + PnL Regression + Queue Toxicity

## Цель
Перевести систему с классификации направления на предсказание Expected PnL с Multimodal Cross-Attention и защитой от Adverse Selection.

## Текущая архитектура
```
OrderbookCNN(2,60→32) + FlowGRU(3,60→32) + Macro(22→64) → 128-dim → MLP(136→6)
Loss: DecoupledTrapLoss = direction CE + confidence MSE + trap BCE
Target: LONG/SHORT/HOLD + confidence + trap_prob
```

## Новая архитектура
```
DeltaBarEncoder(6,100→64) + FlowGRU(3,60→32) + Macro(22→64) 
  → CrossAttention(TradeFlow Q × Orderbook K,V + macro_bias)
  → 128-dim → MLP(136→3)
Loss: AsymmetricPnLLoss = 2.5×(pred-true)² if overestimate, else (pred-true)²
Target: Expected PnL (float) + trap_prob + toxic_flow_prob
```

---

## Компонент 1: 50ms Delta Bars (features.py) ✅ DONE

**Добавлено:**
- `DeltaBar` dataclass: delta_bid_vol, delta_ask_vol, market_buy_vol, market_sell_vol, trade_count, price_velocity
- `_accumulate_trade()` / `_accumulate_ob()` — собирают тики в 50ms бакеты
- `_flush_delta_bar()` — принудительный flush
- `delta_bar_sequence(length=100)` → `(100, 6)` tensor

## Компонент 2: Asymmetric PnL Loss (train.py) ✅ DONE

**Добавлено:**
- `AsymmetricPnLLoss(nn.Module)`: effective_true = pnl - fees, weights = 2.5× if overestimation
- TAKER_FEE = 0.00075, MAKER_FEE = 0.00055

## Компонент 3: Expected PnL Signal (engine.py) — В РАБОТЕ

**Изменения в `_run_tick_prediction`:**
- Добавить `PREDICT_PNL=true` env flag для переключения на新模式
- Новая модель: pred_pnl = logits[0], toxic_prob = sigmoid(logits[1])
- direction = "LONG" if pred_pnl > 0 else "SHORT"
- confidence = min(abs(pred_pnl) / 0.01, 1.0)
- Блокировка при toxic_prob > 0.35 или |pred_pnl| < 0.0015

**Файлы:** `engine.py`, `config.py`

## Компонент 4: Toxic Flow Predictor (nn_models.py + engine.py)

**Новая модель DecisionMLP:**
- out_dim: 6 → 3 (pred_pnl, trap_logit, toxic_logit)
- trap_head + toxic_head на скрытом слое

**В engine.py:** toxic_prob > 0.35 → HOLD

## Компонент 5: Multimodal Cross-Attention (nn_models.py)

**Новый класс:**
- `MultimodalCrossAttention(d_model=64)`: Q=TradeFlow, K=V=Orderbook, + macro_bias
- `DeltaBarEncoder`: Linear(6→64→64) — превращает delta bars в tokens
- Заменяет FlowGRUAttention как контекстный блок
- CNN(Orderbook) → K,V; DeltaBarEncoder → Q
- Выход: context_vector → concat → 128-dim state

## Компонент 6: Go Queue Toxicity Monitor (queue_monitor.go)

**Новый файл:** `oms_execution/internal/executor/queue_monitor.go`
- `QueueMonitor` с activeOrders map и cancelCh
- `MonitorOrder(orderID, symbol, price, side, wallVolume)` — горутина на WebSocket
- `OnOrderbookUpdate(symbol, bids, asks)` — проверка liquidity wall
- Если wall < 25% от initial → экстренный Cancel
- Логирование: "Emergency Cancel: Liquidity wall evaporated"

**Интеграция в service.go:**
- MonitorOrder при placeNewEntry
- RemoveOrder при fill/cancel
- OnOrderbookUpdate из _listen_market_data

---

## Порядок реализации

| # | Компонент | Статус |
|---|-----------|--------|
| 1 | 50ms Delta Bars | ✅ DONE |
| 2 | Asymmetric PnL Loss | ✅ DONE |
| 3 | Expected PnL signal | В РАБОТЕ |
| 4 | Toxic Flow Predictor | ОЖИДАНИЕ |
| 5 | Cross-Attention | ОЖИДАНИЕ |
| 6 | Go Queue Monitor | ОЖИДАНИЕ |

## Совместимость
- state_dim=128 сохраняется (PositionManager compatibility)
- Legacy Conv1d ONNX модели продолжают работать
- Новый режим через PREDICT_PNL=true env flag
