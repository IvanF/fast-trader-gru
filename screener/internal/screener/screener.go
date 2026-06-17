package screener

import (
	"context"
	"log/slog"
	"strconv"
	"time"

	"github.com/fast-trader-gru/screener/internal/bybit"
	"github.com/fast-trader-gru/screener/internal/config"
	"github.com/fast-trader-gru/screener/internal/metrics"
	"github.com/fast-trader-gru/screener/internal/redisx"
)

type SymbolMeta struct {
	Symbol      string  `json:"symbol"`
	FundingRate float64 `json:"funding_rate"`
	Turnover24h float64 `json:"turnover_24h"`
	LastPrice   float64 `json:"last_price"`
}

type ActiveSymbolsPayload struct {
	UpdatedAt int64        `json:"updated_at"`
	Symbols   []string     `json:"symbols"`
	Meta      []SymbolMeta `json:"meta"`
}

type Service struct {
	cfg    config.Config
	bybit  *bybit.Client
	redis  *redisx.Client
	logger *slog.Logger
}

func New(cfg config.Config, bc *bybit.Client, rc *redisx.Client, logger *slog.Logger) *Service {
	return &Service{cfg: cfg, bybit: bc, redis: rc, logger: logger}
}

func (s *Service) Run(ctx context.Context) error {
	if err := s.screenOnce(ctx); err != nil {
		s.logger.Error("initial screen failed", "error", err)
	}

	ticker := time.NewTicker(s.cfg.ScreenInterval)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-ticker.C:
			if err := s.screenOnce(ctx); err != nil {
				s.logger.Error("screen cycle failed", "error", err)
				metrics.BybitRequestErrors.Inc()
			}
		}
	}
}

func (s *Service) screenOnce(ctx context.Context) error {
	start := time.Now()
	defer func() {
		metrics.ScreenDuration.Observe(time.Since(start).Seconds())
	}()

	tickers, err := s.bybit.GetLinearTickers(ctx)
	if err != nil {
		return err
	}

	var active []string
	var meta []SymbolMeta

	for _, t := range tickers {
		if len(t.Symbol) < 4 || t.Symbol[len(t.Symbol)-4:] != "USDT" {
			continue
		}
		if _, blocked := s.cfg.BlacklistSymbols[t.Symbol]; blocked {
			continue
		}

		turnover, err := strconv.ParseFloat(t.Turnover24h, 64)
		if err != nil || turnover < s.cfg.MinTurnover24h {
			metrics.TurnoverFiltered.Inc()
			continue
		}

		fundingRate := 0.0
		if t.FundingRate != "" {
			fundingRate, _ = strconv.ParseFloat(t.FundingRate, 64)
		}

		if fundingRate > s.cfg.MaxFundingRate || fundingRate < s.cfg.MinFundingRate {
			metrics.FundingRateFiltered.Inc()
			s.logger.Debug("funding rate filter rejected symbol",
				"symbol", t.Symbol,
				"funding_rate", fundingRate,
			)
			continue
		}

		lastPrice, _ := strconv.ParseFloat(t.LastPrice, 64)
		active = append(active, t.Symbol)
		meta = append(meta, SymbolMeta{
			Symbol:      t.Symbol,
			FundingRate: fundingRate,
			Turnover24h: turnover,
			LastPrice:   lastPrice,
		})
	}

	payload := ActiveSymbolsPayload{
		UpdatedAt: time.Now().UnixMilli(),
		Symbols:   active,
		Meta:      meta,
	}

	if err := s.redis.Publish(ctx, s.cfg.ActiveSymbolsChannel, payload); err != nil {
		return err
	}
	if err := s.redis.Set(ctx, "config:active_symbols:latest", payload, 0); err != nil {
		s.logger.Warn("failed to cache active symbols", "error", err)
	}

	metrics.ActiveSymbolsCount.Set(float64(len(active)))
	top := active
	if len(top) > 5 {
		top = top[:5]
	}
	s.logger.Info("screener cycle complete",
		"active_symbols", len(active),
		"sample", top,
		"duration_ms", time.Since(start).Milliseconds(),
	)
	return nil
}
