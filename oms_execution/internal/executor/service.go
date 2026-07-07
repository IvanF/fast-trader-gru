package executor

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"math"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/fast-trader-gru/oms_execution/internal/bybit"
	"github.com/fast-trader-gru/oms_execution/internal/codec"
	"github.com/fast-trader-gru/oms_execution/internal/config"
	"github.com/fast-trader-gru/oms_execution/internal/grid"
	"github.com/fast-trader-gru/oms_execution/internal/influx"
	"github.com/fast-trader-gru/oms_execution/internal/liquidity"
	"github.com/fast-trader-gru/oms_execution/internal/metrics"
	"github.com/fast-trader-gru/oms_execution/internal/models"
	"github.com/fast-trader-gru/oms_execution/internal/redisx"
	"github.com/fast-trader-gru/oms_execution/internal/risk"
)

const DynamicConfCap = 0.90

// Funding Rate extremes for bias filter and squeeze detection
const (
	ExtremePositiveFunding = 0.0005  // +0.05% per 8h — longs overleveraged
	ExtremeNegativeFunding = -0.0005 // -0.05% per 8h — shorts overleveraged
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
	_promoting    map[string]bool // symbols currently being promoted from pending → active
	shadow        *ShadowEngine
	priceHistory  map[string][]float64 // symbol -> rolling mid prices for ATR
	volumeHistory map[string][]float64 // symbol -> rolling bid volume for SMA
	symbolStats   map[string]*SymbolStats
	obMomentum    *liquidity.OrderbookMomentum // orderbook pressure velocity tracker
}

type SymbolStats struct {
	TotalTrades  int
	Wins         int
	Losses       int
	ConsecLosses int
	LastPnL      float64
	BannedUntil  time.Time
}

func (ss *SymbolStats) Record(pnl float64) {
	ss.TotalTrades++
	ss.LastPnL = pnl
	if pnl >= 0 {
		ss.Wins++
		ss.ConsecLosses = 0
	} else {
		ss.Losses++
		ss.ConsecLosses++
	}
}

func (ss *SymbolStats) WinRate() float64 {
	if ss.TotalTrades == 0 {
		return 0.5
	}
	return float64(ss.Wins) / float64(ss.TotalTrades)
}

func (ss *SymbolStats) Penalty() float64 {
	if ss.TotalTrades < 3 {
		return 1.0
	}
	wr := ss.WinRate()
	penalty := math.Max(1.0, 3.0-2.5*(wr/0.40))
	if penalty > 3.0 {
		penalty = 3.0
	}
	if ss.ConsecLosses >= 2 {
		streakBonus := 1.0 + float64(ss.ConsecLosses-1)*0.3
		penalty *= streakBonus
		if penalty > 3.75 {
			penalty = 3.75
		}
	}
	return penalty
}

func (ss *SymbolStats) EffectiveConfidence(baseConf float64) float64 {
	effective := baseConf * ss.Penalty()
	if effective > DynamicConfCap {
		effective = DynamicConfCap
	}
	return effective
}

const (
	CircuitBreakerMinTrades = 20
	CircuitBreakerMaxWR     = 0.20
	CircuitBreakerBanHours  = 2
)

func (ss *SymbolStats) IsAutoBanned() bool {
	if !ss.BannedUntil.IsZero() && time.Now().Before(ss.BannedUntil) {
		return true
	}
	if ss.TotalTrades >= CircuitBreakerMinTrades {
		wr := float64(ss.Wins) / float64(ss.TotalTrades)
		if wr < CircuitBreakerMaxWR {
			if ss.BannedUntil.IsZero() || time.Now().After(ss.BannedUntil) {
				ss.BannedUntil = time.Now().Add(time.Duration(CircuitBreakerBanHours) * time.Hour)
			}
			return true
		}
	}
	return false
}

func New(cfg config.Config, bc *bybit.Client, rc *redisx.Client, iw *influx.Writer, logger *slog.Logger) *Service {
	s := &Service{
		cfg:          cfg,
		bybit:        bc,
		redis:        rc,
		influx:       iw,
		logger:       logger,
		tracker:      &metrics.PnLTracker{},
		positions:    make(map[string]*models.ActivePosition),
		pending:      make(map[string]*models.PendingEntry),
		ghostCooldown: make(map[string]int64),
		_promoting:    make(map[string]bool),
		priceHistory:  make(map[string][]float64),
		volumeHistory: make(map[string][]float64),
		symbolStats:   make(map[string]*SymbolStats),
		obMomentum:    liquidity.NewOrderbookMomentum(20),
	}
	if cfg.ShadowMode {
		s.shadow = NewShadowEngine(logger, cfg.ResultsChannel)
		logger.Info("SHADOW MODE ENABLED — no real orders will be placed")
	} else if cfg.ShadowAlwaysEnabled {
		s.shadow = NewShadowEngine(logger, cfg.ResultsChannel)
		logger.Info("SHADOW ALWAYS ENABLED — shadow trades run alongside real execution")
	}
	return s
}

func (s *Service) planOpts() grid.PlanOptions {
	return grid.PlanOptions{
		VolMultiplierCap: s.cfg.VolMultiplierCap,
	}
}

func (s *Service) exitGridOptsForSymbol(symbol string) grid.ExitGridOptions {
	minTP := s.cfg.MinTPPct
	if ob, err := s.redis.GetOrderbook(context.Background(), symbol); err == nil {
		mid := grid.MidPrice(ob)
		if mid > 0 {
			if mid < 0.01 {
				minTP = 0.01
			} else if mid < 0.10 {
				minTP = 0.005
			}
		}
	}
	return grid.ExitGridOptions{
		TPBudgetPct:        s.cfg.TPBudgetPct,
		MinTPPct:           minTP,
		MaxTPPct:           s.cfg.MaxTPPct,
		FeeBreakevenPct:    s.cfg.FeeBreakevenPct,
		MinSLPct:           s.cfg.GetMinSLPct(symbol),
		MaxSLPct:           s.cfg.GetMaxSLPct(symbol),
		TimeStopSec:        s.cfg.GetTimeStopSeconds(symbol),
		EntryFeeRate:       s.cfg.EntryFeeRate,
		ExitFeeRate:        s.cfg.ExitFeeRate,
		TargetNetProfitPct: s.cfg.TargetNetProfitPct,
	}
}

// runPositionManager evaluates ATR-based position management triggers.
func (s *Service) runPositionManager(ctx context.Context, pos *models.ActivePosition, ob models.OrderbookSnapshot) {
	mid := grid.MidPrice(ob)
	if mid <= 0 || pos.OriginalRisk <= 0 {
		return
	}

	// Compute current candle OHLC from available data
	candleHigh := mid
	candleLow := mid
	if len(ob.Bids) > 0 {
		bestBid := grid.MidPrice(models.OrderbookSnapshot{Bids: ob.Bids[:1], Asks: ob.Asks[:1]})
		if bestBid > candleHigh {
			candleHigh = bestBid
		}
		if bestBid < candleLow {
			candleLow = bestBid
		}
	}

	// Volume: use rolling SMA from orderbook history
	var currentVolume, smaVolume float64
	if len(ob.Bids) > 0 {
		for _, lv := range ob.Bids[:min(len(ob.Bids), 10)] {
			var v float64
			fmt.Sscanf(lv.Size, "%f", &v)
			currentVolume += v
		}
	}
	if vh := s.volumeHistory[pos.Symbol]; len(vh) > 0 {
		sum := 0.0
		for _, v := range vh {
			sum += v
		}
		smaVolume = sum / float64(len(vh))
	} else {
		smaVolume = currentVolume
	}

	candleIdx := int(time.Now().Unix() / 60) // Approximate candle index (1-min)

	tradeState := &risk.TradeState{
		EntryPrice:     pos.FillPrice,
		SlPrice:        pos.StopLoss,
		OriginalRisk:   pos.OriginalRisk,
		Direction:      pos.Direction,
		EntryCandleIdx: pos.EntryCandleIdx,
		Size:           pos.RemainingQty,
		InitialSize:    pos.InitialQty,
		ScaledOut:      pos.ScaledOut,
		BreakevenSet:   pos.BreakevenPMSet,
		PriceHistory:   pos.PriceHistory,
	}

	action := risk.ManageOpenTrade(tradeState, mid, candleHigh, candleLow, candleIdx, currentVolume, smaVolume)
	if action == nil {
		return
	}

	switch action.Type {
	case risk.TriggerTimeStop:
		s.logger.Info("PositionManager: Time-Stop triggered",
			"symbol", pos.Symbol, "reason", action.Reason)
		s.timeStopLimitExit(ctx, pos, ob)

	case risk.TriggerScaleOut:
		closeQty := pos.RemainingQty * action.ClosePct
		closeQty = bybit.NormalizeQty(closeQty, pos.QtyStep, pos.MinOrderQty)
		if closeQty > 0 {
			side := closeSide(pos.Direction)
			if _, err := s.bybit.PlaceReduceMarketRetry(ctx, pos.Symbol, side, closeQty, pos.QtyStep); err != nil {
				s.logger.Warn("PositionManager: scale-out failed", "symbol", pos.Symbol, "error", err)
			} else {
				pos.ScaledOut = true
				s.logger.Info("PositionManager: Scale-Out executed",
					"symbol", pos.Symbol, "reason", action.Reason,
					"closed_qty", closeQty,
				)
			}
		}

	case risk.TriggerBreakeven:
		slQty := s.slCoverQty(pos, pos.RemainingQty)
		if slQty > 0 {
			newSL := grid.RoundToTick(action.SlPrice, pos.TickSize)
			if err := s.atomicReplaceStopLoss(ctx, pos, newSL, slQty, "breakeven"); err != nil {
				s.logger.Warn("PositionManager: breakeven failed", "symbol", pos.Symbol, "error", err)
			} else {
				pos.BreakevenPMSet = true
				s.logger.Info("PositionManager: Breakeven set",
					"symbol", pos.Symbol, "reason", action.Reason,
					"new_sl", newSL,
				)
			}
		}

	case risk.TriggerChandelierExit:
		slQty := s.slCoverQty(pos, pos.RemainingQty)
		if slQty > 0 {
			newSL := grid.RoundToTick(action.SlPrice, pos.TickSize)
			if err := s.atomicReplaceStopLoss(ctx, pos, newSL, slQty, "trailing_stop"); err != nil {
				s.logger.Warn("PositionManager: chandelier failed", "symbol", pos.Symbol, "error", err)
			} else {
				s.logger.Info("PositionManager: Chandelier Exit",
					"symbol", pos.Symbol, "reason", action.Reason,
					"new_sl", newSL,
				)
			}
		}
	}
}

func (s *Service) capSignalVol(signal models.TradeSignal) models.TradeSignal {
	signal.VolatilityMultiplier = grid.CapVolMultiplier(signal.VolatilityMultiplier, s.cfg.VolMultiplierCap)
	return signal
}

func (s *Service) Run(ctx context.Context) error {
	go s.runOrderbookCache(ctx)
	if s.cfg.ShadowMode {
		go s.runShadowPriceMonitor(ctx)
	} else {
		go s.runFillMonitor(ctx)
		if s.cfg.ShadowAlwaysEnabled {
			go s.runShadowPriceMonitor(ctx)
		}
	}
	go s.runPositionMonitor(ctx)

	if !s.cfg.ShadowMode {
		s.loadPersistedPositions(ctx)
		s.reconcileOrphanEntryOrders(ctx)
		s.scanOrphanPositions(ctx)
	}

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

	// Shadow-only signal: create shadow trade for training data, no real entry
	if signal.ShadowOnly {
		if s.shadow != nil {
			s.shadowOpen(ctx, signal, recvAt)
		}
		return nil
	}

	if s.cfg.ShadowMode {
		return s.shadowOpen(ctx, signal, recvAt)
	}

	// Shadow always enabled: run shadow alongside real execution
	if s.cfg.ShadowAlwaysEnabled && s.shadow != nil {
		go s.shadowOpen(ctx, signal, recvAt)
	}

	// Guard against TOCTOU: exchange may already have a position while local map is empty.
	s.mu.Lock()
	_, promoting := s._promoting[signal.Symbol]
	s.mu.Unlock()
	if promoting {
		return nil
	}

	// Dynamic confidence: raise threshold for symbols with poor track record
	s.mu.Lock()
	stats := s.symbolStats[signal.Symbol]
	s.mu.Unlock()
	if stats != nil {
		if stats.IsAutoBanned() {
			remaining := time.Until(stats.BannedUntil).Round(time.Minute)
			s.logger.Warn("[CIRCUIT BREAKER] Symbol auto-banned",
				"symbol", signal.Symbol,
				"wr", fmt.Sprintf("%.0f%%", stats.WinRate()*100),
				"total_trades", stats.TotalTrades,
				"remaining", remaining,
			)
			return nil
		}
		effectiveConf := stats.EffectiveConfidence(s.cfg.ConfidenceThreshold)
		if signal.Confidence < effectiveConf {
			s.logger.Info("signal rejected — dynamic confidence",
				"symbol", signal.Symbol,
				"confidence", fmt.Sprintf("%.3f", signal.Confidence),
				"effective_threshold", fmt.Sprintf("%.3f", effectiveConf),
				"wr", fmt.Sprintf("%.0f%%", stats.WinRate()*100),
				"consec_losses", stats.ConsecLosses,
				"penalty", fmt.Sprintf("%.2f", stats.Penalty()),
			)
			return nil
		}
	}

	// Entry quality filters: momentum, volume, spread
	ob, obErr := s.redis.GetOrderbook(ctx, signal.Symbol)
	if obErr == nil {
		mid := grid.MidPrice(ob)
		if mid > 0 {
			if len(ob.Asks) > 0 && len(ob.Bids) > 0 {
				bestBid, _ := strconv.ParseFloat(ob.Bids[0].Price, 64)
				bestAsk, _ := strconv.ParseFloat(ob.Asks[0].Price, 64)
				if bestBid > 0 && bestAsk > 0 {
					spreadPct := (bestAsk - bestBid) / bestBid
					if spreadPct > 0.005 {
						s.logger.Info("signal rejected — spread too wide",
							"symbol", signal.Symbol,
							"spread_pct", fmt.Sprintf("%.4f", spreadPct))
						return nil
					}
				}
			}
			totalBid := 0.0
			for _, l := range ob.Bids[:min(5, len(ob.Bids))] {
				s, _ := strconv.ParseFloat(l.Size, 64)
				totalBid += s
			}
			totalAsk := 0.0
			for _, l := range ob.Asks[:min(5, len(ob.Asks))] {
				s, _ := strconv.ParseFloat(l.Size, 64)
				totalAsk += s
			}
			totalDepth := totalBid + totalAsk
			if totalDepth <= 0 {
				s.logger.Info("signal rejected — zero depth",
					"symbol", signal.Symbol)
				return nil
			}
			bidV := 0.0
			for _, l := range ob.Bids[:min(3, len(ob.Bids))] {
				s, _ := strconv.ParseFloat(l.Size, 64)
				bidV += s
			}
			askV := 0.0
			for _, l := range ob.Asks[:min(3, len(ob.Asks))] {
				s, _ := strconv.ParseFloat(l.Size, 64)
				askV += s
			}
			if bidV+askV > 0 {
				obi := (bidV - askV) / (bidV + askV)

				// Record OBI for momentum tracking
				s.obMomentum.Update(signal.Symbol, ob)
				momentum := s.obMomentum.Momentum(signal.Symbol)
				shift := s.obMomentum.PressureShift(signal.Symbol, 0.3)

				// Static OBI filter: reject if opposite pressure too strong
				if signal.Direction == "SHORT" && obi < -0.4 {
					s.logger.Info("signal rejected — momentum against SHORT",
						"symbol", signal.Symbol,
						"obi", fmt.Sprintf("%.3f", obi),
						"momentum", fmt.Sprintf("%.4f", momentum))
					return nil
				}
				if signal.Direction == "LONG" && obi > 0.4 {
					s.logger.Info("signal rejected — momentum against LONG",
						"symbol", signal.Symbol,
						"obi", fmt.Sprintf("%.3f", obi),
						"momentum", fmt.Sprintf("%.4f", momentum))
					return nil
				}

				// Orderbook momentum filter: reject if pressure is building AGAINST us
				// shift=+1: buying pressure increasing, shift=-1: selling pressure increasing
				if signal.Direction == "LONG" && shift == -1 {
					s.logger.Info("signal rejected — selling pressure building",
						"symbol", signal.Symbol,
						"obi", fmt.Sprintf("%.3f", obi),
						"momentum", fmt.Sprintf("%.4f", momentum))
					return nil
				}
				if signal.Direction == "SHORT" && shift == +1 {
					s.logger.Info("signal rejected — buying pressure building",
						"symbol", signal.Symbol,
						"obi", fmt.Sprintf("%.3f", obi),
						"momentum", fmt.Sprintf("%.4f", momentum))
					return nil
				}

				// Log momentum for analytics (always, not just on rejection)
				if math.Abs(momentum) > 0.1 {
					s.logger.Info("orderbook momentum",
						"symbol", signal.Symbol,
						"obi", fmt.Sprintf("%.3f", obi),
						"momentum", fmt.Sprintf("%.4f", momentum),
						"shift", shift,
						"dir", signal.Direction,
					)
				}
			}
			// Price trend filter: reject if mid moved >0.5% against signal in last 30s
			if hist := s.priceHistory[signal.Symbol]; len(hist) >= 30 {
				recent := hist[len(hist)-30:]
				if len(recent) >= 2 {
					p30 := recent[0]
					pNow := recent[len(recent)-1]
					if p30 > 0 {
						move := (pNow - p30) / p30
						if signal.Direction == "SHORT" && move > 0.005 {
							s.logger.Info("signal rejected — price rising against SHORT",
								"symbol", signal.Symbol, "move_30s", fmt.Sprintf("%.4f", move))
							return nil
						}
						if signal.Direction == "LONG" && move < -0.005 {
							s.logger.Info("signal rejected — price falling against LONG",
								"symbol", signal.Symbol, "move_30s", fmt.Sprintf("%.4f", move))
							return nil
						}
					}
				}
			}

			// Volume filter: reject if volume < 0.5× SMA (dead market)
			if vh := s.volumeHistory[signal.Symbol]; len(vh) >= 5 {
				sum := 0.0
				for _, v := range vh {
					sum += v
				}
				smaVol := sum / float64(len(vh))
				currentVol := 0.0
				for _, lv := range ob.Bids[:min(len(ob.Bids), 10)] {
					var v float64
					fmt.Sscanf(lv.Size, "%f", &v)
					currentVol += v
				}
				if smaVol > 0 && currentVol < smaVol*0.5 {
					s.logger.Info("signal rejected — volume too low",
						"symbol", signal.Symbol,
						"current_vol", fmt.Sprintf("%.1f", currentVol),
						"sma_vol", fmt.Sprintf("%.1f", smaVol))
					return nil
				}
			}
		}

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
	}

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

	// ════════════════════════════════════════════════════════════════
	// CORRELATION FILTER — prevent cluster exposure
	// If we already have 3+ open positions with high BTC correlation,
	// block new entries to prevent correlated macro risk.
	// ════════════════════════════════════════════════════════════════
	const MaxCorrelationThreshold = 0.7
	if signal.BTCorrelation > MaxCorrelationThreshold {
		s.mu.Lock()
		corrPositions := 0
		for _, pos := range s.positions {
			if pos.Signal.BTCorrelation > MaxCorrelationThreshold {
				corrPositions++
			}
		}
		s.mu.Unlock()
		if corrPositions >= 3 {
			s.logger.Warn("signal rejected — cluster correlation risk",
				"symbol", signal.Symbol,
				"btc_correlation", fmt.Sprintf("%.3f", signal.BTCorrelation),
				"corr_positions", corrPositions,
				"limit", 3)
			return nil
		}
	}

	return s.placeNewEntry(ctx, signal, recvAt)
}

// goWithTimeout runs fn in a goroutine that is cancelled after timeoutSec
// seconds or when ctx is done. Prevents goroutine leaks.
func (s *Service) goWithTimeout(ctx context.Context, timeoutSec int, fn func()) {
	go func() {
		timer := time.NewTimer(time.Duration(timeoutSec) * time.Second)
		defer timer.Stop()
		select {
		case <-ctx.Done():
			return
		case <-timer.C:
			fn()
		}
	}()
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

	leverage := s.cfg.GetLeverage(signal.Symbol)
	timeStop := s.cfg.GetTimeStopSeconds(signal.Symbol)

	var qty float64
	var notionalUSD float64
	if s.cfg.UsesUSDSizing() {
		notionalUSD = risk.TradeNotionalUSD(marginUSD, leverage)
		qty = risk.QtyFromNotional(notionalUSD, mid)
	} else {
		qty = s.cfg.DefaultQty
		if signal.PositionScale > 0 {
			qty *= signal.PositionScale
		}
		notionalUSD = qty * mid
		marginUSD = notionalUSD / float64(max(leverage, 1))
	}

	plan := grid.BuildPlan(signal, ob, inst.TickSize, qty, timeStop, s.planOpts())
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

	// ============================================
	// RiskManager: ATR-based SL, EV check, Kelly sizing
	// ============================================
	if s.priceHistory[signal.Symbol] != nil {
		tpPrice := mid
		if len(plan.TakeProfits) > 0 {
			tpPrice = plan.TakeProfits[0]
		}
		riskResult := risk.ProcessSignal(
			plan.Direction,
			signal.Confidence,
			plan.EntryPrice,
			tpPrice,
			s.priceHistory[signal.Symbol],
			s.cfg.AccountDepositUSD,
			signal.VolatilityMultiplier,
			inst.TickSize,
		)
		if !riskResult.Approved {
			s.logger.Info("RiskManager REJECTED",
				"symbol", signal.Symbol,
				"reason", riskResult.RejectReason,
				"sl_dist", fmt.Sprintf("%.3f%%", riskResult.SLDistancePct*100),
				"rr", fmt.Sprintf("%.2f", riskResult.RewardRiskRatio),
				"ev", fmt.Sprintf("%.4f", riskResult.EV),
			)
			return fmt.Errorf("risk rejected: %s", riskResult.RejectReason)
		}
		// Override SL and qty from RiskManager
		plan.StopLoss = riskResult.SlPrice
		if s.cfg.UsesUSDSizing() && riskResult.Qty > 0 {
			riskNotional := riskResult.Qty * mid
			riskMargin := riskNotional / float64(max(leverage, 1))
			if riskMargin <= marginUSD {
				notionalUSD = riskNotional
				qty = riskResult.Qty
				plan.Qty = qty
			}
		}
		s.logger.Info("RiskManager APPROVED",
			"symbol", signal.Symbol,
			"sl", riskResult.SlPrice,
			"rr", fmt.Sprintf("%.2f", riskResult.RewardRiskRatio),
			"ev", fmt.Sprintf("%.4f", riskResult.EV),
			"risk", fmt.Sprintf("%.2f%%", riskResult.RiskPct*100),
			"kelly", fmt.Sprintf("%.2f%%", riskResult.KellyPct*100),
		)
	}

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

	rawQty := plan.Qty
	plan.Qty = bybit.NormalizeQty(plan.Qty, inst.Lot.QtyStep, inst.Lot.MinOrderQty)
	if plan.Qty > inst.Lot.MaxOrderQty {
		return fmt.Errorf("qty %.8f exceeds max %.8f for %s", plan.Qty, inst.Lot.MaxOrderQty, signal.Symbol)
	}

	actualNotional := plan.Qty * mid
	if s.cfg.UsesUSDSizing() {
		actualMargin := actualNotional / float64(max(leverage, 1))
		if actualMargin > marginUSD*1.25 {
			return fmt.Errorf(
				"min lot too large for $%.0f margin budget: need $%.2f margin (qty=%.8f @ %.4f)",
				marginUSD, actualMargin, plan.Qty, mid,
			)
		}
		marginUSD = actualMargin
		if err := s.bybit.SetLeverage(ctx, signal.Symbol, leverage); err != nil {
			s.logger.Warn("set leverage failed", "symbol", signal.Symbol, "leverage", leverage, "error", err)
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

	// ════════════════════════════════════════════════════════════════
	// PASSIVE-AGGRESSIVE ENTRY
	// High-confidence signals (>0.95) → immediate Market-Take
	// Normal signals → Limit Chasing (Maker advantage)
	// ════════════════════════════════════════════════════════════════
	const MarketTakeThreshold = 0.95
	if signal.Confidence >= MarketTakeThreshold {
		placeStart := time.Now()
		orderID, err := s.bybit.PlaceMarketOrder(ctx, plan.Symbol, side, plan.Qty, inst.Lot.QtyStep, false)
		placeDelay := time.Since(placeStart)
		s.logger.Info("Market-Take entry (high confidence)",
			"symbol", signal.Symbol,
			"confidence", fmt.Sprintf("%.3f", signal.Confidence),
			"side", side,
			"qty", plan.Qty,
			"order_id", orderID,
			"place_delay", placeDelay.Round(time.Millisecond),
		)
		if err != nil {
			s.logger.Error("Market-Take failed, falling back to Limit Chasing",
				"symbol", signal.Symbol, "error", err)
		} else {
			pending := &models.PendingEntry{
				Symbol:      plan.Symbol,
				OrderID:     orderID,
				State:       models.PendingEntryStateActive,
				Direction:   plan.Direction,
				Qty:         plan.Qty,
				EntryPrice:  plan.EntryPrice,
				StopLoss:    plan.StopLoss,
				TakeProfits: plan.TakeProfits,
				TimeStopSec: plan.TimeStopSec,
				MarginUSD:   marginUSD,
				NotionalUSD: notionalUSD,
				Leverage:    leverage,
				Signal:      signal,
				QtyStep:     inst.Lot.QtyStep,
				MinOrderQty: inst.Lot.MinOrderQty,
				TickSize:    inst.TickSize,
				PlacedAt:    time.Now().UnixMilli(),
			}
			s.mu.Lock()
			s.pending[plan.Symbol] = pending
			s.mu.Unlock()
			return nil
		}
	}

	const MaxChaseAttempts = 3
	const ChaseTimeoutSec = 5

	// ════════════════════════════════════════════════════════════════
	// VOLATILITY-BASED ENTRY DELAY
	// If price is moving fast (>0.3%/5s), wait for impulse to settle
	// before chasing. Prevents entering at the peak of a spike.
	// ════════════════════════════════════════════════════════════════
	if hist := s.priceHistory[signal.Symbol]; len(hist) >= 10 {
		recent5s := hist[len(hist)-5:]
	older5s := hist[len(hist)-10 : len(hist)-5]
		if len(recent5s) >= 2 && len(older5s) >= 2 {
			pNow := recent5s[len(recent5s)-1]
			pBefore := older5s[0]
			if pBefore > 0 {
				velocity := math.Abs(pNow-pBefore) / pBefore
				if velocity > 0.003 {
					delaySec := 5
					s.logger.Info("volatility entry delay — price moving fast",
						"symbol", signal.Symbol,
						"velocity_5s", fmt.Sprintf("%.4f%%", velocity*100),
						"delay", delaySec)
					time.Sleep(time.Duration(delaySec) * time.Second)
				}
			}
		}
	}

	var orderID string
	for attempt := 1; attempt <= MaxChaseAttempts; attempt++ {
		chaseOb, chaseErr := s.redis.GetOrderbook(ctx, signal.Symbol)
		if chaseErr != nil {
			return fmt.Errorf("chase orderbook: %w", chaseErr)
		}

		var chasePrice float64
		if plan.Direction == "LONG" {
			chasePrice = liquidity.BestBid(chaseOb)
		} else {
			chasePrice = liquidity.BestAsk(chaseOb)
		}
		if chasePrice <= 0 {
			chasePrice = grid.MidPrice(chaseOb)
		}
		if chasePrice <= 0 {
			return fmt.Errorf("chase: no book price for %s", signal.Symbol)
		}
		chasePrice = grid.RoundToTick(chasePrice, inst.TickSize)

		orderID, err = s.bybit.PlaceLimitOrder(ctx, bybit.PlaceOrderRequest{
			Symbol:      plan.Symbol,
			Side:        side,
			Qty:         bybit.FormatQty(plan.Qty, inst.Lot.QtyStep),
			Price:       bybit.FormatPrice(chasePrice),
			PositionIdx: 0,
		})
		if err != nil {
			return fmt.Errorf("chase attempt %d: %w", attempt, err)
		}

		s.logger.Info("Limit Chasing entry",
			"symbol", signal.Symbol,
			"attempt", attempt,
			"price", chasePrice,
			"side", side,
			"qty", plan.Qty,
			"order_id", orderID,
		)

		time.Sleep(time.Duration(ChaseTimeoutSec) * time.Second)

		oi, oiErr := s.bybit.GetOrderRealtime(ctx, signal.Symbol, orderID)
		if oiErr == nil && oi.OrderStatus == "Filled" {
			placeDelay := time.Duration(ChaseTimeoutSec) * time.Second
			s.logger.Info("Limit Chasing filled",
				"symbol", signal.Symbol,
				"attempt", attempt,
				"price", chasePrice,
				"place_delay", placeDelay.Round(time.Millisecond),
			)
			break
		}

		if attempt < MaxChaseAttempts {
			s.logger.Info("Limit Chasing — cancelling unfilled, re-chasing",
				"symbol", signal.Symbol,
				"attempt", attempt,
				"order_status", oi.OrderStatus,
			)
			_ = s.bybit.CancelOrder(ctx, signal.Symbol, orderID)
			orderID = ""
		}
	}

	if orderID == "" {
		return fmt.Errorf("chase failed: %s not filled after %d attempts", signal.Symbol, MaxChaseAttempts)
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
		Leverage:    leverage,
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
	// If pending entry was already removed (e.g. by promotePendingFromExchange),
	// skip the check and proceed directly to position creation.
	if _, still := s.pending[p.Symbol]; still {
		delete(s.pending, p.Symbol)
		s.publishPendingOrder(ctx, "filled", p)
	} else {
		s.publishPendingOrder(ctx, "filled", p)
	}
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
	plannedSL = grid.EnforceMinSLDistance(avgPrice, plannedSL, p.Direction, s.cfg.GetMinSLPct(p.Symbol), p.TickSize)

	entryTime := time.Now().UnixMilli()
	candleIdx := int(entryTime / 60000) // 1-min candle index

	originalRisk := math.Abs(avgPrice - plannedSL)
	if originalRisk <= 0 {
		originalRisk = avgPrice * s.cfg.GetMinSLPct(p.Symbol)
	}

	// Snapshot price history for ATR computation during position lifetime
	var priceSnap []float64
	if hist := s.priceHistory[p.Symbol]; len(hist) > 0 {
		priceSnap = make([]float64, len(hist))
		copy(priceSnap, hist)
	}

	// Detect trading mode dynamically — no hardcoded tickers
	tradingMode := risk.NormalMode
	if len(priceSnap) >= 14 {
		atr := risk.CalculateATR(priceSnap, 14)
		if avgPrice > 0 {
			atrPct := atr / avgPrice
			// Get current spread from orderbook
			var spreadPct float64
			if ob, obErr := s.redis.GetOrderbook(ctx, p.Symbol); obErr == nil {
				if len(ob.Bids) > 0 && len(ob.Asks) > 0 {
					bestBid, _ := strconv.ParseFloat(ob.Bids[0].Price, 64)
					bestAsk, _ := strconv.ParseFloat(ob.Asks[0].Price, 64)
					if bestBid > 0 {
						spreadPct = (bestAsk - bestBid) / bestBid
					}
				}
			}
			tradingMode = risk.DetectTradingMode(atrPct, spreadPct, s.obMomentum.Momentum(p.Symbol), s.cfg.VolatilityThresholdATR, s.cfg.SpreadThresholdPct)
			if tradingMode == risk.HFTScalpingMode {
				s.logger.Warn("[DYNAMIC HFT] volatile instrument detected — HFT Scalping Mode activated",
					"symbol", p.Symbol,
					"atr_pct", fmt.Sprintf("%.4f", atrPct),
					"spread_pct", fmt.Sprintf("%.4f", spreadPct),
					"atr_threshold", fmt.Sprintf("%.4f", s.cfg.VolatilityThresholdATR),
					"spread_threshold", fmt.Sprintf("%.4f", s.cfg.SpreadThresholdPct),
				)
			}
		}
	}

	// Override TimeStop based on trading mode
	timeStopSec := risk.ModeTimeStopSec(tradingMode, s.cfg.TimeStopSeconds, s.cfg.HFTTimeStopSec)

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
		TimeStopSec:  timeStopSec,
		QtyStep:      p.QtyStep,
		MinOrderQty:  p.MinOrderQty,
		TickSize:     p.TickSize,
		MarginUSD:    p.MarginUSD,
		NotionalUSD:  p.NotionalUSD,
		Leverage:     p.Leverage,
		Signal:       p.Signal,
		OrderID:      p.OrderID,
		FilledAt:     entryTime,
		OriginalRisk: originalRisk,
		PriceHistory: priceSnap,
		EntryCandleIdx: candleIdx,
		TradingMode:  int(tradingMode),
	}

	// Register in map before any slow I/O so concurrent signals cannot open a duplicate entry.
	s.mu.Lock()
	if _, exists := s.positions[p.Symbol]; exists {
		// Position already tracked — update with new fill data and ensure exit grid exists
		existing := s.positions[p.Symbol]
		existing.FillPrice = avgPrice
		existing.RemainingQty = qty
		existing.InitialQty = qty
		existing.OrderID = p.OrderID
		existing.OriginalRisk = math.Abs(avgPrice - existing.StopLoss)
		s._promoting[p.Symbol] = false
		s.mu.Unlock()

		// Always deploy exit grid for reprice-halted positions (use existing SL, not pending SL)
		ob2, obErr := s.redis.GetOrderbook(ctx, p.Symbol)
		if obErr == nil && len(ob2.Bids) > 0 {
			_ = s.deployExitGrid(ctx, existing, ob2, avgPrice, existing.StopLoss, p.TickSize)
		}
		s.publishPositionOpened(ctx, existing)
		return
	}
	s.positions[p.Symbol] = pos
	delete(s._promoting, p.Symbol)
	metrics.ActivePositions.Set(float64(len(s.positions)))
	s.mu.Unlock()

	s.persistPosition(ctx, pos)

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

// loadPersistedPositions restores positions from Redis on startup.
// Cross-checks with exchange to remove stale ghost positions.
func (s *Service) loadPersistedPositions(ctx context.Context) {
	positions, err := s.redis.LoadPositions(ctx)
	if err != nil {
		s.logger.Warn("failed to load persisted positions", "error", err)
		return
	}
	if len(positions) == 0 {
		s.logger.Info("no persisted positions to restore")
		return
	}
	s.mu.Lock()
	for _, pos := range positions {
		if _, exists := s.positions[pos.Symbol]; exists {
			continue
		}
		// Verify position still exists on exchange
		exPos, exErr := s.bybit.GetPosition(ctx, pos.Symbol)
		if exErr != nil || exPos.Size <= 0 {
			s.logger.Info("removing stale ghost position from Redis",
				"symbol", pos.Symbol, "exchange_size", exPos.Size)
			go s.removePosition(context.Background(), pos.Symbol)
			continue
		}
		pos.ExitGridReady = false
		s.positions[pos.Symbol] = pos
		metrics.ActivePositions.Set(float64(len(s.positions)))
	}
	s.mu.Unlock()
	s.logger.Info("restored persisted positions", "count", len(s.positions))
}

// persistPosition saves a position to Redis (fire-and-forget).
func (s *Service) persistPosition(ctx context.Context, pos *models.ActivePosition) {
	if err := s.redis.SavePosition(ctx, pos); err != nil {
		s.logger.Warn("position persist failed", "symbol", pos.Symbol, "error", err)
	}
}

// removePosition deletes a position from Redis (fire-and-forget).
func (s *Service) removePosition(ctx context.Context, symbol string) {
	if err := s.redis.DeletePosition(ctx, symbol); err != nil {
		s.logger.Warn("position remove failed", "symbol", symbol, "error", err)
	}
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
			// Update price history for ATR computation
			mid := grid.MidPrice(ob)
			if mid > 0 {
				hist := s.priceHistory[ob.Symbol]
				hist = append(hist, mid)
				if len(hist) > 100 {
					hist = hist[len(hist)-100:]
				}
				s.priceHistory[ob.Symbol] = hist
			}
			// Update volume history for PositionManager SMA
			var bidVol float64
			for _, lv := range ob.Bids[:min(len(ob.Bids), 10)] {
				var v float64
				fmt.Sscanf(lv.Size, "%f", &v)
				bidVol += v
			}
			if bidVol > 0 {
				vh := s.volumeHistory[ob.Symbol]
				vh = append(vh, bidVol)
				if len(vh) > 20 {
					vh = vh[len(vh)-20:]
				}
				s.volumeHistory[ob.Symbol] = vh
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
		notional := exSize * pos.FillPrice
		if len(pos.TakeProfitOrders) == 0 && notional < 15.0 {
			s.logger.Warn("grid failed with small remainder, flattening",
				"symbol", symbol, "qty", exSize, "notional", notional)
			_ = s.bybit.CancelAllOrders(ctx, pos.Symbol)
			if err := s.ensureExchangeFlat(ctx, pos, "grid_fail_remainder_close"); err != nil {
				s.logger.Warn("grid fail flatten failed", "symbol", symbol, "error", err)
			} else {
				s.tryFinalizePosition(ctx, pos, "grid_fail_close", 0)
			}
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

	// Close small remainder BEFORE retryMissingTakeProfits creates new TPs
	closeSmall := false
	if len(pos.TakeProfitOrders) == 0 && !pos.PartialTaken {
		closeSmall = true
		s.logger.Warn("position has NO TPs",
			"symbol", pos.Symbol, "qty", exSize, "notional", exSize*pos.FillPrice, "direction", pos.Direction)
	}
	// Safety net: if grid was deployed but all TPs lost — force redeploy
	if !closeSmall && pos.ExitGridReady && len(pos.TakeProfitOrders) == 0 && exSize > pos.MinOrderQty {
		s.logger.Warn("TPs lost on active position, forcing redeploy",
			"symbol", pos.Symbol, "qty", exSize, "notional", exSize*pos.FillPrice)
		pos.ExitGridReady = false
		pos.GridDeployFailures = 0
	}
	if !closeSmall {
		allTPsFilled := true
		for _, tp := range pos.TakeProfitOrders {
			if !tp.Filled {
				allTPsFilled = false
				break
			}
		}
		if allTPsFilled && pos.PartialTaken {
			closeSmall = true
		}
	}
	if closeSmall {
		exSize, hasPos, _ := s.syncPositionFromExchange(ctx, pos)
		if hasPos && exSize > 0 {
			notional := exSize * pos.FillPrice
			s.logger.Info("remainder check",
				"symbol", pos.Symbol, "qty", exSize, "notional", notional, "close", notional < 20.0)
			if notional < 20.0 {
				s.logger.Warn("closing small remainder",
					"symbol", pos.Symbol, "qty", exSize, "notional", notional)
				s.cancelExitOrders(ctx, pos)
				if err := s.ensureExchangeFlat(ctx, pos, "remainder_close"); err != nil {
					s.logger.Warn("remainder close failed", "symbol", pos.Symbol, "error", err)
				} else {
					s.tryFinalizePosition(ctx, pos, "take_profit_grid", 0)
					return
				}
			} else {
				s.logger.Warn("position WITHOUT TP but notional too large to close, triggering redeploy",
					"symbol", pos.Symbol, "qty", exSize, "notional", notional)
				closeSmall = false
				pos.ExitGridReady = false
				pos.GridDeployFailures = 0
			}
		}
	}

	s.retryMissingTakeProfits(ctx, pos, ob)

	// HARD TIME-STOP: 2-stage FSM — PostOnly → Kill-Switch after 30s
	// Timing adapts to trading mode: HFT gets shorter windows
	// Also adapts to signal confidence: high-conf gets longer runway
	mode := risk.TradingMode(pos.TradingMode)
	normalTimeStop := s.cfg.TimeStopSeconds
	hftTimeStop := s.cfg.HFTTimeStopSec
	// Dynamic: high-confidence signals get 600s, low-confidence get 180s
	if pos.Signal.Confidence >= 0.90 {
		normalTimeStop = 240
	} else {
		normalTimeStop = 120
	}
	hardTimeStopSec := risk.ModeTimeStopSec(mode, normalTimeStop, hftTimeStop)
	const MakerFillTimeoutSec = 30
	const ZombieRetrySec = 120
	holdSec := (time.Now().UnixMilli() - pos.EntryTime) / 1000
	if holdSec > int64(hardTimeStopSec) {
		exSize, hasPos, _ := s.syncPositionFromExchange(ctx, pos)
		if hasPos && exSize > 0 {
			// ════════════════════════════════════════════════════════════════
			// FIX 3: BREAKEVEN EXIT — if in profit at time-stop, close at breakeven
			// instead of taking market loss. Protects profitable trades from
			// being killed by time-stop.
			// ════════════════════════════════════════════════════════════════
			mid := grid.MidPrice(ob)
			if mid > 0 {
				commissionBuffer := pos.FillPrice * 0.0013
				var isProfitable bool
				if pos.Direction == "LONG" {
					isProfitable = mid > pos.FillPrice+commissionBuffer
				} else {
					isProfitable = mid < pos.FillPrice-commissionBuffer
				}
				if isProfitable && !pos.TimeStopPlaced {
					s.logger.Info("[TIME-STOP FIX] Breakeven exit — position in profit",
						"symbol", pos.Symbol,
						"direction", pos.Direction,
						"entry", pos.FillPrice,
						"current", mid,
						"hold_sec", holdSec)
					s.cancelExitOrders(ctx, pos)
					side := "Buy"
					if pos.Direction == "LONG" {
						side = "Sell"
					}
					normQty := bybit.NormalizeQty(exSize, pos.QtyStep, pos.MinOrderQty)
					_, err := s.bybit.PlaceReduceMarketRetry(ctx, pos.Symbol, side, normQty, pos.QtyStep)
					if err != nil {
						s.logger.Error("breakeven exit failed", "symbol", pos.Symbol, "error", err)
					}
					pos.TimeStopPlaced = true
					s.tryFinalizePosition(ctx, pos, "time_stop", mid)
					return
				}
			}

			// ════════════════════════════════════════════════════════════════
			// FIX 5: LONG PENALTY — for LONG positions with big loss at time-stop,
			// force immediate market close to limit damage.
			// ════════════════════════════════════════════════════════════════
			if pos.Direction == "LONG" && !pos.TimeStopPlaced {
				if mid > 0 {
					longLoss := (pos.FillPrice - mid) / pos.FillPrice
					if longLoss > 0.003 { // > 0.3% loss
						s.logger.Warn("[TIME-STOP FIX] LONG penalty — force immediate close",
							"symbol", pos.Symbol,
							"loss_pct", fmt.Sprintf("%.2f%%", longLoss*100),
							"hold_sec", holdSec)
						s.cancelExitOrders(ctx, pos)
						normQty := bybit.NormalizeQty(exSize, pos.QtyStep, pos.MinOrderQty)
						_, err := s.bybit.PlaceReduceMarketRetry(ctx, pos.Symbol, "Sell", normQty, pos.QtyStep)
						if err != nil {
							s.logger.Error("LONG penalty close failed", "symbol", pos.Symbol, "error", err)
						}
						pos.TimeStopPlaced = true
						s.tryFinalizePosition(ctx, pos, "time_stop", mid)
						return
					}
				}
			}

			if !pos.TimeStopPlaced {
				s.logger.Warn("HARD TIME-STOP — initiating 2-stage exit",
					"symbol", pos.Symbol,
					"hold_sec", holdSec,
					"qty", exSize,
				)
				s.cancelExitOrders(ctx, pos)

				side := "Buy"
				if pos.Direction == "LONG" {
					side = "Sell"
				}
				normQty := bybit.NormalizeQty(exSize, pos.QtyStep, pos.MinOrderQty)

				exitPrice := grid.PassiveMakerExitPrice(pos.Direction, ob, pos.TickSize, s.cfg.EntryMakerTicks)
				if exitPrice > 0 {
					orderID, orderErr := s.bybit.PlaceReducePostOnlyLimit(ctx, pos.Symbol, side, normQty, pos.QtyStep, bybit.FormatPrice(exitPrice))
					if orderErr == nil {
						pos.TimeStopPlaced = true
						s.logger.Info("HARD TIME-STOP stage 1 — PostOnly Maker order placed",
							"symbol", pos.Symbol, "price", exitPrice, "qty", normQty, "order_id", orderID)

						s.goWithTimeout(ctx, MakerFillTimeoutSec, func() {
							s.mu.Lock()
							currentPos, exists := s.positions[pos.Symbol]
							if !exists || currentPos == nil || !currentPos.TimeStopPlaced {
								s.mu.Unlock()
								return
							}
							s.mu.Unlock()

							oi, oiErr := s.bybit.GetOrderRealtime(ctx, pos.Symbol, orderID)
							if oiErr == nil && oi.OrderStatus == "New" {
								s.logger.Warn("HARD TIME-STOP stage 2 — Market Kill-Switch!",
									"symbol", pos.Symbol,
									"order_id", orderID)
								_ = s.bybit.CancelOrder(ctx, pos.Symbol, orderID)

								_, merr := s.bybit.PlaceReduceMarketRetry(ctx, pos.Symbol, side, normQty, pos.QtyStep)
								if merr != nil {
									s.logger.Error("Market Kill-Switch failed",
										"symbol", pos.Symbol, "error", merr)
								} else {
									s.logger.Info("Market Kill-Switch executed",
										"symbol", pos.Symbol)
		}
	}

	// ════════════════════════════════════════════════════════════════
	// FIX 4: TRAILING TIME-STOP — if position reached 50% of TP
	// distance, move SL to breakeven immediately (don't wait for
	// breakevenSec timer). Protects profits from time-stop reversal.
	// ════════════════════════════════════════════════════════════════
	if !pos.BreakevenSet && holdSec > 60 {
		mid := grid.MidPrice(ob)
		if mid > 0 && len(pos.TakeProfitOrders) > 0 {
			tpPrice := pos.TakeProfitOrders[0].Price
			if tpPrice > 0 {
				riskDist := math.Abs(pos.FillPrice - pos.StopLoss)
				if riskDist > 0 {
					profitPct := math.Abs(mid - pos.FillPrice)
					halfTP := math.Abs(tpPrice-pos.FillPrice) * 0.5
					if profitPct >= halfTP {
						commissionBuffer := pos.FillPrice * 0.0013
						var trailingBE float64
						if pos.Direction == "LONG" {
							trailingBE = pos.FillPrice + commissionBuffer
						} else {
							trailingBE = pos.FillPrice - commissionBuffer
						}
						trailingBE = math.Round(trailingBE/pos.TickSize) * pos.TickSize

						if pos.Direction == "LONG" && trailingBE > pos.StopLoss ||
							pos.Direction == "SHORT" && trailingBE < pos.StopLoss {
							exSize, hasPos, _ := s.syncPositionFromExchange(ctx, pos)
							if hasPos && exSize > 0 {
								slQty := s.slCoverQty(pos, exSize)
								if slQty > 0 {
									if err := s.atomicReplaceStopLoss(ctx, pos, trailingBE, slQty, "trailing_breakeven"); err != nil {
										s.logger.Warn("trailing breakeven failed",
											"symbol", pos.Symbol, "error", err)
									} else {
										pos.BreakevenSet = true
										s.logger.Info("[TIME-STOP FIX] Trailing breakeven — 50% TP reached",
											"symbol", pos.Symbol,
											"direction", pos.Direction,
											"entry", pos.FillPrice,
											"current", mid,
											"tp", tpPrice,
											"trailing_be", trailingBE,
											"hold_sec", holdSec,
										)
									}
								}
							}
						}
					}
				}
			}
		}
	}
						})
						return
					}
					s.logger.Warn("HARD TIME-STOP — PostOnly failed, immediate market Kill-Switch",
						"symbol", pos.Symbol, "error", orderErr)
				}

				s.logger.Warn("HARD TIME-STOP — immediate market Kill-Switch",
					"symbol", pos.Symbol, "qty", normQty)
				_, mktErr := s.bybit.PlaceReduceMarketRetry(ctx, pos.Symbol, side, normQty, pos.QtyStep)
				if mktErr != nil {
					s.logger.Error("Market Kill-Switch failed",
						"symbol", pos.Symbol, "error", mktErr)
				}
				pos.TimeStopPlaced = true
				return
			}
			// ZOMBIE RETRY: if TimeStopPlaced but still open after ZombieRetrySec, force market
			if pos.TimeStopPlaced && holdSec > int64(hardTimeStopSec)+ZombieRetrySec {
				s.logger.Error("ZOMBIE DETECTED — force market close",
					"symbol", pos.Symbol, "hold_sec", holdSec, "qty", exSize)
				pos.TimeStopPlaced = false
				side := "Buy"
				if pos.Direction == "LONG" {
					side = "Sell"
				}
				normQty := bybit.NormalizeQty(exSize, pos.QtyStep, pos.MinOrderQty)
				_, mktErr := s.bybit.PlaceReduceMarketRetry(ctx, pos.Symbol, side, normQty, pos.QtyStep)
				if mktErr != nil {
					s.logger.Error("ZOMBIE force close failed", "symbol", pos.Symbol, "error", mktErr)
				} else {
					s.logger.Info("ZOMBIE force close executed", "symbol", pos.Symbol)
					pos.TimeStopPlaced = true
				}
				return
			}
		}
		if !hasPos {
			s.tryFinalizePosition(ctx, pos, "time_stop", 0)
			return
		}
	}

	if s.monitorExitOrders(ctx, pos) {
		return
	}

	// BREAKEVEN: move SL to breakeven after 60 seconds
	if holdSec > 60 && !pos.BreakevenSet {
		mid := grid.MidPrice(ob)
		if mid > 0 {
			commissionBuffer := pos.FillPrice * 0.0013 // 0.13% covers both sides
			var breakevenPrice float64
			if pos.Direction == "LONG" {
				breakevenPrice = pos.FillPrice + commissionBuffer
			} else {
				breakevenPrice = pos.FillPrice - commissionBuffer
			}
			breakevenPrice = math.Round(breakevenPrice/pos.TickSize) * pos.TickSize

			// Only tighten (never widen)
			if pos.Direction == "LONG" && breakevenPrice > pos.StopLoss ||
				pos.Direction == "SHORT" && breakevenPrice < pos.StopLoss {
				exSize, hasPos, _ := s.syncPositionFromExchange(ctx, pos)
				if hasPos && exSize > 0 {
					slQty := s.slCoverQty(pos, exSize)
					if slQty > 0 {
						if err := s.atomicReplaceStopLoss(ctx, pos, breakevenPrice, slQty, "breakeven"); err != nil {
				s.logger.Warn("180s breakeven SL replace failed",
					"symbol", pos.Symbol, "error", err)
						} else {
							pos.BreakevenSet = true
							s.logger.Info("180s breakeven SL set",
								"symbol", pos.Symbol,
								"breakeven", breakevenPrice,
								"fill", pos.FillPrice,
								"hold_sec", holdSec,
							)
						}
					}
				}
			} else {
				pos.BreakevenSet = true // Already at or past breakeven
			}
		}
	}

	// PositionManager: ATR-based triggers (scale-out, breakeven, chandelier)
	s.runPositionManager(ctx, pos, ob)

	// ABSOLUTE SAFETY NET: runs AFTER all other logic, guarantees every open position has TPs
	activeTPs := 0
	for _, tp := range pos.TakeProfitOrders {
		if !tp.Filled {
			activeTPs++
		}
	}
	if activeTPs > 0 {
		s.maybeRefreshExitGrid(ctx, pos, ob)
		return
	}

	// 0 active TPs on open position — need action
	notional := exSize * pos.FillPrice
	if notional < 10.0 {
		// Too small for TP grid — close immediately
		s.logger.Warn("safety net: position too small to manage, flattening",
			"symbol", pos.Symbol, "qty", exSize, "notional", notional)
		s.cancelExitOrders(ctx, pos)
		if err := s.ensureExchangeFlat(ctx, pos, "safety_net_too_small"); err != nil {
			s.logger.Warn("safety net flatten failed", "symbol", pos.Symbol, "error", err)
		} else {
			s.tryFinalizePosition(ctx, pos, "safety_net_close", 0)
			return
		}
	} else {
		// Big enough for TP grid — deploy now
		s.logger.Warn("safety net: deploying TP grid for unprotected position",
			"symbol", pos.Symbol, "qty", exSize, "notional", notional)
		pos.ExitGridReady = false
		pos.GridDeployFailures = 0
		if err := s.deployExitGrid(ctx, pos, ob, pos.PlannedEntry, pos.PlannedSL, pos.TickSize); err != nil {
			s.logger.Warn("safety net deploy failed", "symbol", pos.Symbol, "error", err)
			pos.GridDeployFailures++
			pos.LastGridDeployFailure = time.Now().UnixMilli()
		}
	}
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

		closeReason := "exchange_closed"

		// Hard time-stop: if position was closed by our time-stop logic, use that reason
		if pos.TimeStopPlaced {
			closeReason = "time_stop"
		}

		for _, tp := range pos.TakeProfitOrders {
			if tp.Price > 0 {
				tpDist := math.Abs(closed.AvgExitPrice-tp.Price) / tp.Price
				if tpDist < 0.002 {
					closeReason = "take_profit"
					s.logger.Info("TP triggered (exit≈tp)",
						"symbol", pos.Symbol,
						"exit", closed.AvgExitPrice,
						"tp", tp.Price,
						"tp_dist", tpDist,
					)
					break
				}
			}
		}
		if closeReason == "exchange_closed" {
			if pos.StopLossOrder != nil && pos.StopLossOrder.Price > 0 {
				slDist := math.Abs(closed.AvgExitPrice-pos.StopLossOrder.Price) / pos.StopLossOrder.Price
				if slDist < 0.002 {
					closeReason = "stop_loss"
					s.logger.Info("SL triggered (exit≈sl)",
						"symbol", pos.Symbol,
						"exit", closed.AvgExitPrice,
						"sl", pos.StopLossOrder.Price,
						"sl_dist", slDist,
					)
				}
			}
			if closeReason == "exchange_closed" && pos.StopLoss > 0 {
				slDist := math.Abs(closed.AvgExitPrice-pos.StopLoss) / pos.StopLoss
				if slDist < 0.002 {
					closeReason = "stop_loss"
					s.logger.Info("SL triggered (exit≈sl)",
						"symbol", pos.Symbol,
						"exit", closed.AvgExitPrice,
						"sl", pos.StopLoss,
						"sl_dist", slDist,
					)
				}
			}
		}

		s.finalizeClose(ctx, pos, closed.ClosedPnL, closed.AvgEntryPrice, closed.AvgExitPrice, closeReason, true)
		s.mu.Lock()
		delete(s.positions, pos.Symbol)
		s.ghostCooldown[pos.Symbol] = time.Now().UnixMilli()+120_000
		metrics.ActivePositions.Set(float64(len(s.positions)))
		metrics.GridActive.WithLabelValues(pos.Symbol).Set(0)
		s.mu.Unlock()
		s.removePosition(ctx, pos.Symbol)
		return
	}

	s.finalizeClose(ctx, pos, 0, pos.FillPrice, pos.FillPrice, "stale_tracker_removed", false)

	s.mu.Lock()
	s.ghostCooldown[pos.Symbol] = time.Now().UnixMilli()+120_000
	s.mu.Unlock()

	holdSec := (time.Now().UnixMilli() - pos.EntryTime) / 1000
	orderDelay := 0.0
	if pos.FilledAt > 0 {
		orderDelay = float64(time.Now().UnixMilli()-pos.FilledAt) / 1000.0
	}
	s.logger.Warn("stale position tracker removed",
		"symbol", pos.Symbol,
		"hold_sec", holdSec,
		"order_placement_delay", fmt.Sprintf("%.1fs", orderDelay),
		"has_exit_orders", len(pos.TakeProfitOrders) > 0 || pos.StopLossOrder != nil,
		"reason", "no fill on exit orders — position was stuck")
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
	stats := s.symbolStats[pos.Symbol]
	if stats == nil {
		stats = &SymbolStats{}
		s.symbolStats[pos.Symbol] = stats
	}
	stats.Record(pnl)
	metrics.ActivePositions.Set(float64(len(s.positions)))
	s.mu.Unlock()
	s.removePosition(ctx, pos.Symbol)

	s.logger.Info("position closed",
		"symbol", pos.Symbol,
		"direction", pos.Direction,
		"reason", reason,
		"pnl", pnl,
		"exchange_pnl", exchangePnL,
		"entry", entryPrice,
		"exit", exitPrice,
		"hold_ms", hold.Milliseconds(),
		"symbol_wr", fmt.Sprintf("%.0f%%", stats.WinRate()*100),
		"consec_losses", stats.ConsecLosses,
		"penalty", fmt.Sprintf("%.2f", stats.Penalty()),
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

func (s *Service) shadowOpen(ctx context.Context, signal models.TradeSignal, recvAt time.Time) error {
	signal = s.capSignalVol(signal)
	marginUSD := s.cfg.TradeMarginUSD

	ob, err := s.redis.GetOrderbook(ctx, signal.Symbol)
	if err != nil {
		s.logger.Warn("shadow: no orderbook", "symbol", signal.Symbol)
		return nil
	}

	mid := grid.MidPrice(ob)
	if mid <= 0 {
		return nil
	}

	plan := grid.BuildPlan(signal, ob, 0.001, 0, s.cfg.TimeStopSeconds, s.planOpts())
	entry := plan.EntryPrice
	if entry <= 0 {
		entry = mid
	}

	var qty float64
	if s.cfg.UsesUSDSizing() {
		qty = math.Round(marginUSD*float64(s.cfg.Leverage)/mid*1000) / 1000
	} else {
		qty = s.cfg.DefaultQty
	}
	if qty <= 0 {
		return nil
	}

	exitOpts := s.exitGridOptsForSymbol(signal.Symbol)
	exitGrid := grid.BuildExitGrid(
		signal.Direction, entry, entry, plan.StopLoss,
		ob, signal, 0.001, qty, 0.001, 0.001, exitOpts,
	)

	tpPrice := entry
	if len(exitGrid.TakeProfits) > 0 {
		tpPrice = exitGrid.TakeProfits[0].Price
	}

	s.shadow.OpenPosition(signal, entry, exitGrid.StopLoss.Price, tpPrice, qty)

	s.logger.Info("SHADOW: entry simulated",
		"symbol", signal.Symbol, "direction", signal.Direction,
		"entry", entry, "sl", exitGrid.StopLoss.Price, "tp", tpPrice,
		"qty", qty, "conf", signal.Confidence,
	)

	return nil
}

func (s *Service) runShadowPriceMonitor(ctx context.Context) {
	ticker := time.NewTicker(500 * time.Millisecond)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			if s.shadow.ActiveCount() == 0 {
				continue
			}
			for _, sym := range s.shadow.Symbols() {
				ob, err := s.redis.GetOrderbook(ctx, sym)
				if err != nil {
					continue
				}
				mid := grid.MidPrice(ob)
				if mid > 0 {
					s.shadow.ProcessPriceUpdate(ctx, sym, mid, func(ctx context.Context, channel string, msg interface{}) {
						_ = s.redis.Publish(ctx, channel, msg)
					})
				}
			}
		}
	}
}
