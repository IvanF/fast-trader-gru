package executor

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"strings"
	"sync"
	"time"

	"github.com/fast-trader-gru/oms_execution/internal/bybit"
	"github.com/fast-trader-gru/oms_execution/internal/codec"
	"github.com/fast-trader-gru/oms_execution/internal/config"
	"github.com/fast-trader-gru/oms_execution/internal/grid"
	"github.com/fast-trader-gru/oms_execution/internal/influx"
	"github.com/fast-trader-gru/oms_execution/internal/metrics"
	"github.com/fast-trader-gru/oms_execution/internal/models"
	"github.com/fast-trader-gru/oms_execution/internal/redisx"
	"github.com/fast-trader-gru/oms_execution/internal/risk"
)

type Service struct {
	cfg       config.Config
	bybit     *bybit.Client
	redis     *redisx.Client
	influx    *influx.Writer
	logger    *slog.Logger
	tracker   *metrics.PnLTracker
	mu            sync.Mutex
	deployMu      sync.Mutex
	gridDeploying map[string]bool
	positions     map[string]*models.ActivePosition
	pending   map[string]*models.PendingEntry
	ghostCooldown map[string]int64
}

func New(cfg config.Config, bc *bybit.Client, rc *redisx.Client, iw *influx.Writer, logger *slog.Logger) *Service {
	return &Service{
		cfg:          cfg,
		bybit:        bc,
		redis:        rc,
		influx:       iw,
		logger:       logger,
		tracker:      &metrics.PnLTracker{},
		positions:    make(map[string]*models.ActivePosition),
		pending:      make(map[string]*models.PendingEntry),
		ghostCooldown: make(map[string]int64),
	}
}

func (s *Service) planOpts() grid.PlanOptions {
	return grid.PlanOptions{
		EntryMakerTicks:  s.cfg.EntryMakerTicks,
		VolMultiplierCap: s.cfg.VolMultiplierCap,
	}
}

func (s *Service) exitGridOpts() grid.ExitGridOptions {
	return grid.ExitGridOptions{
		TPBudgetPct:     s.cfg.TPBudgetPct,
		MinTPPct:        s.cfg.MinTPPct,
		MaxTPPct:        s.cfg.MaxTPPct,
		FeeBreakevenPct: s.cfg.FeeBreakevenPct,
		MinSLPct:        s.cfg.MinSLPct,
	}
}

func (s *Service) capSignalVol(signal models.TradeSignal) models.TradeSignal {
	signal.VolatilityMultiplier = grid.CapVolMultiplier(signal.VolatilityMultiplier, s.cfg.VolMultiplierCap)
	return signal
}

func (s *Service) Run(ctx context.Context) error {
	go s.runOrderbookCache(ctx)
	go s.runFillMonitor(ctx)
	go s.runPositionMonitor(ctx)

	s.reconcileOrphanEntryOrders(ctx)
	// Adopt any open exchange positions missed before this OMS session started.
	s.scanOrphanPositions(ctx)

	for {
		if ctx.Err() != nil {
			return ctx.Err()
		}
		pubsub := s.redis.Subscribe(ctx, s.cfg.SignalsChannel)
		ch := pubsub.Channel()
		s.logger.Info("subscribed to signals", "channel", s.cfg.SignalsChannel)

		for msg := range ch {
			recvAt := time.Now()
			var signal models.TradeSignal
			if err := json.Unmarshal([]byte(msg.Payload), &signal); err != nil {
				s.logger.Warn("invalid signal", "error", err)
				continue
			}
			if signal.Direction == "HOLD" {
				continue
			}
			if err := s.handleSignal(ctx, signal, recvAt); err != nil {
				s.logger.Error("signal handling failed", "symbol", signal.Symbol, "error", err)
			}
		}

		if ctx.Err() != nil {
			pubsub.Close()
			return ctx.Err()
		}
		s.logger.Warn("signal subscription dropped, reconnecting")
		pubsub.Close()
		time.Sleep(2 * time.Second)
	}
}

func (s *Service) usedMarginUSD() float64 {
	var used float64
	for _, p := range s.positions {
		used += p.MarginUSD
	}
	for _, p := range s.pending {
		used += p.MarginUSD
	}
	return used
}

func (s *Service) handleSignal(ctx context.Context, signal models.TradeSignal, recvAt time.Time) error {
	s.mu.Lock()
	_, hasPos := s.positions[signal.Symbol]
	_, hasPending := s.pending[signal.Symbol]
	s.mu.Unlock()

	if signal.SetupAction == "abort_setup" {
		s.mu.Lock()
		p, ok := s.pending[signal.Symbol]
		s.mu.Unlock()
		if ok {
			reason := "abort_setup"
			if signal.AbortReason != "" {
				reason = signal.AbortReason
			}
			s.cancelPendingEntry(ctx, p, reason)
		}
		return nil
	}

	if signal.ExitReason != "" && !hasPos {
		return nil
	}

	if hasPos || hasPending {
		return s.reconcileExistingSignal(ctx, signal, recvAt)
	}

	// Guard against TOCTOU: exchange may already have a position while local map is empty.
	exPos, err := s.bybit.GetPosition(ctx, signal.Symbol)
	if err == nil && exPos.Size > 0 {
		dir := "LONG"
		if exPos.Side == "Sell" {
			dir = "SHORT"
		}
		if signal.Direction == dir {
			s.logger.Info("exchange position exists without local tracker, adopting",
				"symbol", signal.Symbol,
				"size", exPos.Size,
				"direction", dir,
			)
			s.adoptExchangePosition(ctx, exPos, "signal_adopt")
			return s.reconcileExistingSignal(ctx, signal, recvAt)
		}
		s.logger.Info("skipping entry — exchange position opposite direction",
			"symbol", signal.Symbol,
			"exchange", dir,
			"signal", signal.Direction,
		)
		return nil
	}

	return s.placeNewEntry(ctx, signal, recvAt)
}

func (s *Service) placeNewEntry(ctx context.Context, signal models.TradeSignal, recvAt time.Time) error {
	signal = s.capSignalVol(signal)
	s.mu.Lock()
	if p, ok := s.pending[signal.Symbol]; ok {
		if p.State == models.PendingEntryStateCancelling {
			s.mu.Unlock()
			s.logger.Debug("placeNewEntry blocked — cancel in flight", "symbol", signal.Symbol)
			return nil
		}
	}
	marginUSD := s.cfg.TradeMarginUSD
	if !s.cfg.UsesUSDSizing() {
		marginUSD = 0
	} else if s.usedMarginUSD()+marginUSD > s.cfg.AccountDepositUSD {
		s.mu.Unlock()
		return fmt.Errorf(
			"margin budget exceeded: used=%.2f + trade=%.2f > deposit=%.2f",
			s.usedMarginUSD(), marginUSD, s.cfg.AccountDepositUSD,
		)
	}
	s.mu.Unlock()

	ob, err := s.redis.GetOrderbook(ctx, signal.Symbol)
	if err != nil {
		return fmt.Errorf("orderbook: %w", err)
	}

	inst, err := s.bybit.GetInstrument(ctx, signal.Symbol)
	if err != nil {
		s.logger.Warn("instrument info fallback", "symbol", signal.Symbol, "error", err)
		inst = bybit.InstrumentInfo{
			TickSize: 0.01,
			Lot:      bybit.LotFilters{QtyStep: 0.001, MinOrderQty: 0.001, MaxOrderQty: 1e9},
		}
	}

	mid := grid.MidPrice(ob)
	if mid <= 0 {
		return fmt.Errorf("invalid mark price for %s", signal.Symbol)
	}

	var qty float64
	var notionalUSD float64
	if s.cfg.UsesUSDSizing() {
		notionalUSD = risk.TradeNotionalUSD(marginUSD, s.cfg.Leverage)
		qty = risk.QtyFromNotional(notionalUSD, mid)
	} else {
		qty = s.cfg.DefaultQty
		if signal.PositionScale > 0 {
			qty *= signal.PositionScale
		}
		notionalUSD = qty * mid
		marginUSD = notionalUSD / float64(max(s.cfg.Leverage, 1))
	}

	plan := grid.BuildPlan(signal, ob, inst.TickSize, qty, s.cfg.TimeStopSeconds, s.planOpts())
	if signal.StopLoss > 0 {
		plan.StopLoss = signal.StopLoss
	}
	if len(signal.TakeProfits) > 0 {
		plan.TakeProfits = signal.TakeProfits
	}
	// Prefer live spread-edge maker price; fall back to ML anchor when maker ticks disabled.
	if s.cfg.EntryMakerTicks > 0 {
		if makerEntry := grid.AggressiveMakerEntry(plan.Direction, ob, inst.TickSize, s.cfg.EntryMakerTicks); makerEntry > 0 {
			plan.EntryPrice = makerEntry
		}
	} else if signal.EntryPrice > 0 {
		plan.EntryPrice = signal.EntryPrice
	}
	rawQty := plan.Qty
	plan.Qty = bybit.NormalizeQty(plan.Qty, inst.Lot.QtyStep, inst.Lot.MinOrderQty)
	if plan.Qty > inst.Lot.MaxOrderQty {
		return fmt.Errorf("qty %.8f exceeds max %.8f for %s", plan.Qty, inst.Lot.MaxOrderQty, signal.Symbol)
	}

	actualNotional := plan.Qty * mid
	if s.cfg.UsesUSDSizing() {
		actualMargin := actualNotional / float64(max(s.cfg.Leverage, 1))
		if actualMargin > marginUSD*1.25 {
			return fmt.Errorf(
				"min lot too large for $%.0f margin budget: need $%.2f margin (qty=%.8f @ %.4f)",
				marginUSD, actualMargin, plan.Qty, mid,
			)
		}
		marginUSD = actualMargin
		if err := s.bybit.SetLeverage(ctx, signal.Symbol, s.cfg.Leverage); err != nil {
			s.logger.Warn("set leverage failed", "symbol", signal.Symbol, "leverage", s.cfg.Leverage, "error", err)
		}
	}

	if plan.Qty != rawQty {
		s.logger.Info("order qty adjusted to exchange minimum",
			"symbol", signal.Symbol,
			"requested", rawQty,
			"normalized", plan.Qty,
			"min", inst.Lot.MinOrderQty,
			"step", inst.Lot.QtyStep,
		)
	}

	metrics.FundingRate.WithLabelValues(signal.Symbol).Set(signal.FundingRate)

	side := "Buy"
	if plan.Direction == "SHORT" {
		side = "Sell"
	}

	orderID, err := s.bybit.PlaceLimitOrder(ctx, bybit.PlaceOrderRequest{
		Symbol:      plan.Symbol,
		Side:        side,
		Qty:         bybit.FormatQty(plan.Qty, inst.Lot.QtyStep),
		Price:       bybit.FormatPrice(plan.EntryPrice),
		PositionIdx: 0,
	})
	if err != nil {
		return err
	}

	metrics.SignalToOrderLatency.Observe(time.Since(recvAt).Seconds())
	metrics.OrdersPlaced.WithLabelValues("entry").Inc()

	pending := &models.PendingEntry{
		Symbol:      plan.Symbol,
		OrderID:     orderID,
		State:       models.PendingEntryStateActive,
		Direction:   plan.Direction,
		EntryPrice:  plan.EntryPrice,
		StopLoss:    plan.StopLoss,
		TakeProfits: plan.TakeProfits,
		Qty:         plan.Qty,
		TimeStopSec: plan.TimeStopSec,
		QtyStep:     inst.Lot.QtyStep,
		MinOrderQty: inst.Lot.MinOrderQty,
		TickSize:    inst.TickSize,
		MarginUSD:   marginUSD,
		NotionalUSD: actualNotional,
		Leverage:    s.cfg.Leverage,
		Signal:      signal,
		PlacedAt:    time.Now().UnixMilli(),
		Orderbook:   ob,
	}

	s.mu.Lock()
	s.pending[plan.Symbol] = pending
	s.mu.Unlock()

	s.publishPendingOrder(ctx, "placed", pending)

	s.logger.Info("entry order placed, waiting for fill",
		"symbol", plan.Symbol,
		"direction", plan.Direction,
		"order_id", orderID,
		"entry", plan.EntryPrice,
		"qty", plan.Qty,
	)
	return nil
}

func (s *Service) runFillMonitor(ctx context.Context) {
	ticker := time.NewTicker(500 * time.Millisecond)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			s.mu.Lock()
			pendingCopy := make([]*models.PendingEntry, 0, len(s.pending))
			for _, p := range s.pending {
				pendingCopy = append(pendingCopy, p)
			}
			s.mu.Unlock()

			for _, p := range pendingCopy {
				s.checkPendingFill(ctx, p)
			}
		}
	}
}

func (s *Service) checkPendingFill(ctx context.Context, p *models.PendingEntry) {
	if s.pendingEntryIsCancelling(p) {
		return
	}
	ob, err := s.redis.GetOrderbook(ctx, p.Symbol)
	if err == nil {
		mid := grid.MidPrice(ob)
		if s.pendingEntryStale(p, mid) {
			s.cancelPendingEntry(ctx, p, "entry_stale_market_moved")
			return
		}
		s.maybePegPendingEntry(ctx, p, ob)
	}

	timeoutMs := int64(s.cfg.OrderFillTimeoutSec) * 1000
	if timeoutMs > 0 && time.Now().UnixMilli()-p.PlacedAt > timeoutMs {
		s.cancelPendingEntry(ctx, p, "fill_timeout")
		return
	}

	order, err := s.bybit.GetOrderRealtime(ctx, p.Symbol, p.OrderID)
	if err != nil {
		s.logger.Warn("order status check failed", "symbol", p.Symbol, "error", err)
		return
	}

	if order.OrderStatus == "Filled" {
		s.finalizePendingEntry(ctx, p)
		return
	}

	// Partial fill: do not promote until fully filled; stale/confidence cancel handles remainder.
	if order.OrderStatus == "PartiallyFilled" && order.CumExecQty > 0 {
		s.mu.Lock()
		_, hasPos := s.positions[p.Symbol]
		s.mu.Unlock()
		if hasPos {
			s.syncActiveWithExchange(ctx, p.Symbol)
		}
	}
}

func (s *Service) finalizePendingEntry(ctx context.Context, p *models.PendingEntry) {
	_ = s.bybit.CancelOrder(ctx, p.Symbol, p.OrderID)

	exPos, err := s.bybit.GetPosition(ctx, p.Symbol)
	if err != nil || exPos.Size <= 0 || !s.positionMatchesDirection(exPos, p.Direction) {
		s.mu.Lock()
		delete(s.pending, p.Symbol)
		s.mu.Unlock()
		return
	}

	avgPrice := exPos.AvgPrice
	if avgPrice <= 0 {
		avgPrice = p.EntryPrice
	}
	qty := bybit.NormalizeQty(exPos.Size, p.QtyStep, p.MinOrderQty)
	s.promotePending(ctx, p, avgPrice, qty)
}

func (s *Service) cancelPendingEntry(ctx context.Context, p *models.PendingEntry, reason string) {
	if !s.beginPendingEntryCancel(p) {
		s.logger.Debug("cancel skipped — already in flight", "symbol", p.Symbol, "reason", reason)
		return
	}
	defer s.releasePendingEntryCancelIfStuck(p)

	outcome, cancelErr := s.cancelEntryOrderConfirmed(ctx, p, p.OrderID)
	if outcome == entryOrderExecuted {
		if s.promotePendingFromExchange(ctx, p, reason+"_filled_on_cancel") {
			return
		}
	}

	if exPos := s.pollPositionAfterCancel(ctx, p.Symbol, p.Direction, cancelErr); exPos != nil {
		if s.tryPromotePendingFromExchange(ctx, p, reason, exPos) {
			return
		}
	}

	if cancelErr != nil && outcome != entryOrderExecuted {
		s.logger.Warn("cancel pending order failed", "symbol", p.Symbol, "order_id", p.OrderID, "error", cancelErr)
	}

	s.mu.Lock()
	delete(s.pending, p.Symbol)
	s.mu.Unlock()
	s.publishPendingCancelled(ctx, p, reason)
	s.logger.Info("pending entry cancelled", "symbol", p.Symbol, "reason", reason, "order_id", p.OrderID)
}

func (s *Service) syncActiveWithExchange(ctx context.Context, symbol string) {
	s.mu.Lock()
	pos, ok := s.positions[symbol]
	s.mu.Unlock()
	if !ok {
		return
	}
	exSize, hasPos, err := s.syncPositionFromExchange(ctx, pos)
	if err != nil || !hasPos {
		return
	}
	pos.InitialQty = exSize
	pos.RemainingQty = exSize
	s.checkPositionSizeViolation(ctx, pos, exSize)
	s.syncSLToFullRemainder(ctx, pos)
}

func (s *Service) positionMatchesDirection(exPos bybit.PositionInfo, direction string) bool {
	if direction == "LONG" {
		return exPos.Side == "Buy"
	}
	return exPos.Side == "Sell"
}

func (s *Service) promotePending(ctx context.Context, p *models.PendingEntry, avgPrice, qty float64) {
	s.mu.Lock()
	if _, still := s.pending[p.Symbol]; !still {
		s.mu.Unlock()
		return
	}
	delete(s.pending, p.Symbol)
	s.publishPendingOrder(ctx, "filled", p)
	if existing, ok := s.positions[p.Symbol]; ok {
		s.mu.Unlock()
		s.logger.Warn("promote skipped — position already tracked, syncing exchange size",
			"symbol", p.Symbol,
			"existing_qty", existing.RemainingQty,
			"requested_qty", qty,
		)
		s.syncActiveWithExchange(ctx, p.Symbol)
		return
	}
	s.mu.Unlock()

	_ = s.bybit.CancelOrder(ctx, p.Symbol, p.OrderID)

	exPos, err := s.bybit.GetPosition(ctx, p.Symbol)
	if err == nil && exPos.Size > 0 && s.positionMatchesDirection(exPos, p.Direction) {
		if exPos.AvgPrice > 0 {
			avgPrice = exPos.AvgPrice
		}
		qty = bybit.NormalizeQty(exPos.Size, p.QtyStep, p.MinOrderQty)
	}
	if qty <= 0 {
		return
	}

	ob := p.Orderbook
	if fresh, err := s.redis.GetOrderbook(ctx, p.Symbol); err == nil {
		ob = fresh
	}

	plannedSL := p.StopLoss
	if plannedSL <= 0 {
		plan := grid.BuildPlan(p.Signal, ob, p.TickSize, qty, p.TimeStopSec, s.planOpts())
		plannedSL = plan.StopLoss
	}
	plannedSL = grid.EnforceMinSLDistance(avgPrice, plannedSL, p.Direction, s.cfg.MinSLPct, p.TickSize)

	entryTime := time.Now().UnixMilli()
	pos := &models.ActivePosition{
		Symbol:       p.Symbol,
		Direction:    p.Direction,
		FillPrice:    avgPrice,
		PlannedEntry: p.EntryPrice,
		PlannedSL:    plannedSL,
		TargetQty:    p.Qty,
		InitialQty:   qty,
		RemainingQty: qty,
		StopLoss:     plannedSL,
		EntryTime:    entryTime,
		TimeStopSec:  p.TimeStopSec,
		QtyStep:      p.QtyStep,
		MinOrderQty:  p.MinOrderQty,
		TickSize:     p.TickSize,
		MarginUSD:    p.MarginUSD,
		NotionalUSD:  p.NotionalUSD,
		Leverage:     p.Leverage,
		Signal:       p.Signal,
		OrderID:      p.OrderID,
		FilledAt:     entryTime,
	}

	// Register in map before any slow I/O so concurrent signals cannot open a duplicate entry.
	s.mu.Lock()
	if _, exists := s.positions[p.Symbol]; exists {
		s.mu.Unlock()
		s.syncActiveWithExchange(ctx, p.Symbol)
		return
	}
	s.positions[p.Symbol] = pos
	metrics.ActivePositions.Set(float64(len(s.positions)))
	s.mu.Unlock()

	metrics.GridActive.WithLabelValues(p.Symbol).Set(1)
	s.logger.Info("position opened (exchange fill confirmed)",
		"symbol", p.Symbol,
		"direction", p.Direction,
		"fill_price", avgPrice,
		"qty", qty,
		"target_qty", p.Qty,
		"order_id", p.OrderID,
	)

	if s.checkPositionSizeViolation(ctx, pos, qty) {
		if pos.RemainingQty > pos.MinOrderQty*0.99 {
			ob, err := s.redis.GetOrderbook(ctx, p.Symbol)
			if err == nil {
				_ = s.deployExitGrid(ctx, pos, ob, p.EntryPrice, plannedSL, p.TickSize)
			}
		}
		s.publishPositionOpened(ctx, pos)
		return
	}

	if err := s.deployExitGrid(ctx, pos, ob, p.EntryPrice, plannedSL, p.TickSize); err != nil {
		s.logger.Error("exit grid deploy failed", "symbol", p.Symbol, "error", err)
	}
	s.publishPositionOpened(ctx, pos)
}

func (s *Service) cancelPending(ctx context.Context, p *models.PendingEntry, reason string) {
	s.cancelPendingEntry(ctx, p, reason)
}

func (s *Service) runOrderbookCache(ctx context.Context) {
	for {
		if ctx.Err() != nil {
			return
		}
		pubsub := s.redis.PSubscribe(ctx, "market:orderbook:*")
		ch := pubsub.Channel()
		s.logger.Info("subscribed to orderbook cache", "pattern", "market:orderbook:*")
		for msg := range ch {
			var ob models.OrderbookSnapshot
			if err := codec.Unmarshal([]byte(msg.Payload), &ob); err != nil {
				s.logger.Warn("orderbook cache decode failed", "channel", msg.Channel, "error", err)
				continue
			}
			if ob.Symbol == "" {
				ob.Symbol = symbolFromChannel(msg.Channel)
			}
			if ob.Symbol == "" || len(ob.Bids) == 0 || len(ob.Asks) == 0 {
				continue
			}
			if err := s.redis.SetOrderbook(ctx, ob.Symbol, ob); err != nil {
				s.logger.Warn("orderbook cache write failed", "symbol", ob.Symbol, "error", err)
			}
		}
		if ctx.Err() != nil {
			pubsub.Close()
			return
		}
		s.logger.Warn("orderbook cache subscription dropped, reconnecting")
		pubsub.Close()
		time.Sleep(2 * time.Second)
	}
}

func symbolFromChannel(channel string) string {
	parts := strings.Split(channel, ":")
	if len(parts) >= 3 {
		return parts[2]
	}
	return ""
}

func (s *Service) runPositionMonitor(ctx context.Context) {
	ticker := time.NewTicker(500 * time.Millisecond)
	orphanTicker := time.NewTicker(2 * time.Second)
	defer ticker.Stop()
	defer orphanTicker.Stop()

	for {
		select {
		case <-ctx.Done():
			return
		case <-orphanTicker.C:
			s.scanOrphanPositions(ctx)
		case <-ticker.C:
			s.mu.Lock()
			symbols := make([]string, 0, len(s.positions))
			for sym := range s.positions {
				symbols = append(symbols, sym)
			}
			s.mu.Unlock()

			for _, sym := range symbols {
				s.evaluatePosition(ctx, sym)
			}
		}
	}
}

func (s *Service) evaluatePosition(ctx context.Context, symbol string) {
	s.mu.Lock()
	pos, ok := s.positions[symbol]
	if !ok {
		s.mu.Unlock()
		return
	}
	s.mu.Unlock()

	exSize, hasPos, err := s.syncPositionFromExchange(ctx, pos)
	if err != nil {
		return
	}
	if !hasPos || exSize <= 0 {
		s.handleGhostPosition(ctx, pos)
		return
	}
	if s.checkPositionSizeViolation(ctx, pos, exSize) {
		return
	}

	if !pos.ExitGridReady {
		if pos.GridDeployFailures >= 3 {
			s.logger.Warn("exit grid exhausted retries, force-flattening", "symbol", symbol, "failures", pos.GridDeployFailures)
			s.handleGhostPosition(ctx, pos)
			return
		}
		if pos.LastGridDeployFailure > 0 && time.Now().UnixMilli()-pos.LastGridDeployFailure < 2_000 {
			return
		}
		if exSize <= pos.MinOrderQty*1.01 {
			s.logger.Warn("position at min qty, force flattening via market", "symbol", symbol, "qty", exSize)
			_ = s.bybit.CancelAllOrders(ctx, pos.Symbol)
			side := closeSide(pos.Direction)
			_, _ = s.bybit.PlaceReduceMarketRetry(ctx, pos.Symbol, side, exSize, pos.QtyStep)
			s.handleGhostPosition(ctx, pos)
			return
		}
		ob, err := s.redis.GetOrderbook(ctx, symbol)
		if err != nil {
			return
		}
		if err := s.deployExitGrid(ctx, pos, ob, pos.PlannedEntry, pos.PlannedSL, pos.TickSize); err != nil {
			pos.GridDeployFailures++
			pos.LastGridDeployFailure = time.Now().UnixMilli()
			s.logger.Warn("exit grid retry failed", "symbol", symbol, "error", err, "failures", pos.GridDeployFailures)
		} else {
			pos.GridDeployFailures = 0
		}
		return
	}

	ob, err := s.redis.GetOrderbook(ctx, symbol)
	if err != nil {
		return
	}
	s.retryMissingTakeProfits(ctx, pos, ob)

	if elapsedMs(pos.EntryTime) > int64(pos.TimeStopSec)*1000 {
		// Infrastructure GC only — normal exits are event-driven (alpha decay, TP/SL grid).
		s.timeStopLimitExit(ctx, pos, ob)
		return
	}

	if s.monitorExitOrders(ctx, pos) {
		return
	}

	s.maybeRefreshExitGrid(ctx, pos, ob)
}

func (s *Service) handleGhostPosition(ctx context.Context, pos *models.ActivePosition) {
	if err := s.ensureExchangeFlat(ctx, pos, "ghost_position"); err != nil {
		s.logger.Warn("ghost flatten failed", "symbol", pos.Symbol, "error", err)
	}

	closed, err := s.bybit.GetRecentClosedPnL(ctx, pos.Symbol, pos.EntryTime)
	if err == nil && closed != nil && closed.UpdatedTime >= pos.EntryTime {
		exSize, hasPos, _ := s.syncPositionFromExchange(ctx, pos)
		if hasPos && exSize > 0 {
			s.logger.Warn("ghost position still on exchange after flatten, force removing tracker",
				"symbol", pos.Symbol, "ex_size", exSize,
			)
		}
		s.finalizeClose(ctx, pos, closed.ClosedPnL, closed.AvgEntryPrice, closed.AvgExitPrice, "exchange_closed", true)
		s.mu.Lock()
		delete(s.positions, pos.Symbol)
		s.ghostCooldown[pos.Symbol] = time.Now().UnixMilli()+120_000
		metrics.ActivePositions.Set(float64(len(s.positions)))
		metrics.GridActive.WithLabelValues(pos.Symbol).Set(0)
		s.mu.Unlock()
		return
	}

	s.cleanupSymbolOrdersAfterClose(ctx, pos.Symbol, pos)

	s.mu.Lock()
	delete(s.positions, pos.Symbol)
	s.ghostCooldown[pos.Symbol] = time.Now().UnixMilli()+120_000
	metrics.ActivePositions.Set(float64(len(s.positions)))
	metrics.GridActive.WithLabelValues(pos.Symbol).Set(0)
	s.mu.Unlock()
	s.logger.Info("removed stale position tracker", "symbol", pos.Symbol)
}

func (s *Service) finalizeClose(
	ctx context.Context,
	pos *models.ActivePosition,
	pnl, entryPrice, exitPrice float64,
	reason string,
	exchangePnL bool,
) {
	s.cleanupSymbolOrdersAfterClose(ctx, pos.Symbol, pos)

	hold := time.Duration(time.Now().UnixMilli()-pos.EntryTime) * time.Millisecond
	s.tracker.Record(pnl, hold)
	metrics.SymbolPnL.WithLabelValues(pos.Symbol).Add(pnl)
	metrics.OrdersPlaced.WithLabelValues("close").Inc()
	metrics.GridActive.WithLabelValues(pos.Symbol).Set(0)

	result := models.ExecutionResult{
		SignalID:      pos.Signal.SignalID,
		Symbol:        pos.Symbol,
		Direction:     pos.Direction,
		StateVector:   pos.Signal.StateVector,
		EntryPrice:    entryPrice,
		ExitPrice:     exitPrice,
		NetPnL:        pnl,
		HoldingTimeMs: hold.Milliseconds(),
		Regime:        pos.Signal.Regime,
		ClosedAt:      time.Now().UnixMilli(),
		PartialClosed: pos.PartialTaken,
		GridLevels:    len(pos.TakeProfitOrders),
		CloseReason:   reason,
		ExchangePnL:   exchangePnL,
	}
	_ = s.redis.Publish(ctx, s.cfg.ResultsChannel, result)
	if s.influx != nil {
		s.influx.WriteTradeOutcome(result)
	}

	s.mu.Lock()
	delete(s.positions, pos.Symbol)
	metrics.ActivePositions.Set(float64(len(s.positions)))
	s.mu.Unlock()

	s.logger.Info("position closed",
		"symbol", pos.Symbol,
		"reason", reason,
		"pnl", pnl,
		"exchange_pnl", exchangePnL,
		"entry", entryPrice,
		"exit", exitPrice,
		"hold_ms", hold.Milliseconds(),
	)
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}

func minFloat(a, b float64) float64 {
	if a < b {
		return a
	}
	return b
}
