package executor

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"net/http"
	"time"

	"github.com/fast-trader-gru/oms_execution/internal/bybit"
	"github.com/fast-trader-gru/oms_execution/internal/grid"
	"github.com/fast-trader-gru/oms_execution/internal/metrics"
	"github.com/fast-trader-gru/oms_execution/internal/models"
)

func (s *Service) scanOrphanPositions(ctx context.Context) {
	seen := make(map[string]bool)

	positions, err := s.bybit.ListOpenPositions(ctx)
	if err != nil {
		s.logger.Warn("orphan scan list failed", "error", err)
	} else {
		for _, exPos := range positions {
			seen[exPos.Symbol] = true
			s.tryAdoptOrphan(ctx, exPos)
		}
	}

	symbols, err := s.redis.GetActiveSymbols(ctx)
	if err != nil {
		s.logger.Warn("orphan scan active symbols failed", "error", err)
		return
	}
	for _, sym := range symbols {
		if seen[sym] {
			continue
		}
		exPos, err := s.bybit.GetPosition(ctx, sym)
		if err != nil || exPos.Size <= 0 {
			continue
		}
		s.tryAdoptOrphan(ctx, exPos)
	}
}

func (s *Service) tryAdoptOrphan(ctx context.Context, exPos bybit.PositionInfo) {
	s.mu.Lock()
	_, tracked := s.positions[exPos.Symbol]
	_, pending := s.pending[exPos.Symbol]
	until, inCooldown := s.ghostCooldown[exPos.Symbol]
	s.mu.Unlock()
	if tracked || pending {
		return
	}
	if inCooldown && time.Now().UnixMilli() < until {
		return
	}
	if inCooldown {
		s.mu.Lock()
		delete(s.ghostCooldown, exPos.Symbol)
		s.mu.Unlock()
	}
	s.adoptExchangePosition(ctx, exPos, "orphan_scan")
}

func (s *Service) requestExitPlanFromML(ctx context.Context, symbol, direction string, ob models.OrderbookSnapshot, tickSize float64) map[string]interface{} {
	mlURL := "http://ftg-ml-engine:10103/exit-plan"
	s.logger.Info("requesting exit plan from ML", "symbol", symbol, "direction", direction)

	type orderbookLevel struct {
		Price string `json:"price"`
		Size  string `json:"size"`
	}
	type exitPlanRequest struct {
		Direction     string           `json:"direction"`
		Bids          []orderbookLevel `json:"bids"`
		Asks          []orderbookLevel `json:"asks"`
		TickSize      float64          `json:"tick_size"`
		VolMultiplier float64          `json:"volatility_multiplier"`
		Regime        string           `json:"regime"`
		Confidence    float64          `json:"confidence"`
	}

	req := exitPlanRequest{
		Direction:     direction,
		TickSize:      tickSize,
		VolMultiplier: 1.0,
		Regime:        "Choppy",
		Confidence:    s.cfg.ConfidenceThreshold,
	}
	for _, l := range ob.Bids {
		req.Bids = append(req.Bids, orderbookLevel{Price: l.Price, Size: l.Size})
	}
	for _, l := range ob.Asks {
		req.Asks = append(req.Asks, orderbookLevel{Price: l.Price, Size: l.Size})
	}

	body, _ := json.Marshal(req)
	httpCtx, cancel := context.WithTimeout(ctx, 2*time.Second)
	defer cancel()
	httpReq, err := http.NewRequestWithContext(httpCtx, "POST", mlURL, bytes.NewReader(body))
	if err != nil {
		s.logger.Warn("exit-plan request failed", "symbol", symbol, "error", err)
		return nil
	}
	httpReq.Header.Set("Content-Type", "application/json")

	resp, err := http.DefaultClient.Do(httpReq)
	if err != nil {
		s.logger.Warn("exit-plan request failed", "symbol", symbol, "error", err)
		return nil
	}
	defer resp.Body.Close()

	respBody, _ := io.ReadAll(resp.Body)
	if resp.StatusCode != 200 {
		s.logger.Warn("exit-plan request returned non-200", "symbol", symbol, "status", resp.StatusCode)
		return nil
	}

	var result map[string]interface{}
	if err := json.Unmarshal(respBody, &result); err != nil {
		s.logger.Warn("exit-plan response parse failed", "symbol", symbol, "error", err)
		return nil
	}
	if _, ok := result["error"]; ok {
		s.logger.Warn("exit-plan returned error", "symbol", symbol, "error", result["error"])
		return nil
	}
	return result
}

func (s *Service) adoptExchangePosition(ctx context.Context, exPos bybit.PositionInfo, reason string) {
	direction := "LONG"
	switch exPos.Side {
	case "Sell":
		direction = "SHORT"
	case "Buy":
		direction = "LONG"
	default:
		s.logger.Warn("orphan position unknown side, skipping", "symbol", exPos.Symbol, "side", exPos.Side)
		return
	}
	qty := exPos.Size
	fillPrice := exPos.AvgPrice
	if fillPrice <= 0 {
		return
	}

	inst, err := s.bybit.GetInstrument(ctx, exPos.Symbol)
	if err != nil {
		inst = bybit.InstrumentInfo{
			TickSize: 0.0001,
			Lot:      bybit.LotFilters{QtyStep: 0.1, MinOrderQty: 0.1, MaxOrderQty: 1e9},
		}
	}
	qty = bybit.NormalizeQty(qty, inst.Lot.QtyStep, inst.Lot.MinOrderQty)
	if qty <= 0 {
		return
	}

	signal := models.TradeSignal{
		Symbol:               exPos.Symbol,
		Direction:            direction,
		Confidence:           s.cfg.ConfidenceThreshold,
		VolatilityMultiplier: 1.0,
		Regime:               "Choppy",
		Timestamp:            time.Now().UnixMilli(),
	}

	ob, err := s.redis.GetOrderbook(ctx, exPos.Symbol)
	if err != nil {
		ob = models.OrderbookSnapshot{Symbol: exPos.Symbol}
	}

	leverage := s.cfg.GetLeverage(exPos.Symbol)
	timeStop := s.cfg.GetTimeStopSeconds(exPos.Symbol)
	minSLPct := s.cfg.GetMinSLPct(exPos.Symbol)

	plan := grid.BuildPlan(signal, ob, inst.TickSize, qty, timeStop, s.planOpts())
	if plan.StopLoss <= 0 {
		plan.StopLoss = fillPrice * 0.99
		if direction == "SHORT" {
			plan.StopLoss = fillPrice * 1.01
		}
	}
	plan.StopLoss = grid.EnforceMinSLDistance(fillPrice, plan.StopLoss, direction, minSLPct, inst.TickSize)

	if exitPlan := s.requestExitPlanFromML(context.Background(), exPos.Symbol, direction, ob, inst.TickSize); exitPlan != nil {
		if sl, ok := exitPlan["stop_loss"].(float64); ok && sl > 0 {
			plan.StopLoss = sl
			signal.StopLoss = sl
		}
		if tps, ok := exitPlan["take_profits"].([]interface{}); ok && len(tps) > 0 {
			floatTPs := make([]float64, 0, len(tps))
			for _, tp := range tps {
				if f, ok := tp.(float64); ok && f > 0 {
					floatTPs = append(floatTPs, f)
				}
			}
			if len(floatTPs) > 0 {
				signal.TakeProfits = floatTPs
				plan.TakeProfits = floatTPs
				s.logger.Info("orphan exit plan from ML",
					"symbol", exPos.Symbol, "tps", floatTPs, "sl", plan.StopLoss)
			}
		}
	}

	notional := qty * fillPrice
	marginUSD := notional / float64(max(leverage, 1))

	pos := &models.ActivePosition{
		Symbol:       exPos.Symbol,
		Direction:    direction,
		FillPrice:    fillPrice,
		PlannedEntry: plan.EntryPrice,
		PlannedSL:    plan.StopLoss,
		InitialQty:   qty,
		TargetQty:    qty,
		RemainingQty: qty,
		StopLoss:     plan.StopLoss,
		EntryTime:    time.Now().UnixMilli(),
		TimeStopSec:  timeStop,
		QtyStep:      inst.Lot.QtyStep,
		MinOrderQty:  inst.Lot.MinOrderQty,
		TickSize:     inst.TickSize,
		MarginUSD:    marginUSD,
		NotionalUSD:  notional,
		Leverage:     leverage,
		Signal:       signal,
		FilledAt:     time.Now().UnixMilli(),
	}

	s.mu.Lock()
	if _, exists := s.positions[exPos.Symbol]; exists {
		s.mu.Unlock()
		return
	}
	s.positions[exPos.Symbol] = pos
	metrics.ActivePositions.Set(float64(len(s.positions)))
	metrics.GridActive.WithLabelValues(exPos.Symbol).Set(1)
	s.mu.Unlock()

	s.logger.Warn("adopted orphan exchange position",
		"symbol", exPos.Symbol,
		"reason", reason,
		"direction", direction,
		"fill_price", fillPrice,
		"qty", qty,
		"planned_sl", plan.StopLoss,
	)

	if err := s.deployExitGrid(ctx, pos, ob, plan.EntryPrice, plan.StopLoss, inst.TickSize); err != nil {
		notional := qty * fillPrice
		if len(pos.TakeProfitOrders) == 0 && notional < 15.0 {
			s.logger.Warn("orphan with no TPs and small notional, flattening",
				"symbol", exPos.Symbol, "qty", qty, "notional", notional)
			s.cancelExitOrders(ctx, pos)
			if flatErr := s.ensureExchangeFlat(ctx, pos, "orphan_remainder_close"); flatErr != nil {
				s.logger.Error("orphan flatten failed", "symbol", exPos.Symbol, "error", flatErr)
			} else {
				s.tryFinalizePosition(ctx, pos, "orphan_remainder_close", 0)
				return
			}
		}
		s.logger.Error("orphan exit grid deploy failed", "symbol", exPos.Symbol, "error", err)
	}
	s.publishPositionOpened(ctx, pos)
}
