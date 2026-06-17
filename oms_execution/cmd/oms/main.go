package main

import (
	"context"
	"log/slog"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/fast-trader-gru/oms_execution/internal/bybit"
	"github.com/fast-trader-gru/oms_execution/internal/config"
	"github.com/fast-trader-gru/oms_execution/internal/executor"
	"github.com/fast-trader-gru/oms_execution/internal/influx"
	"github.com/fast-trader-gru/oms_execution/internal/metrics"
	"github.com/fast-trader-gru/oms_execution/internal/redisx"
	"github.com/fast-trader-gru/oms_execution/internal/risk"
)

func main() {
	logger := slog.New(slog.NewJSONHandler(os.Stdout, &slog.HandlerOptions{Level: slog.LevelInfo}))
	cfg := config.Load()

	if cfg.BybitAPIKey == "" || cfg.BybitAPISecret == "" {
		logger.Warn("BYBIT_API_KEY / BYBIT_API_SECRET not set — orders will fail against live API")
	}

	ctx, cancel := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)
	defer cancel()

	go func() {
		mux := http.NewServeMux()
		mux.Handle("/metrics", metrics.Handler())
		mux.HandleFunc("/healthz", func(w http.ResponseWriter, _ *http.Request) {
			w.WriteHeader(http.StatusOK)
			_, _ = w.Write([]byte("ok"))
		})
		logger.Info("metrics server listening", "addr", cfg.MetricsAddr)
		if err := http.ListenAndServe(cfg.MetricsAddr, mux); err != nil && err != http.ErrServerClosed {
			logger.Error("metrics server failed", "error", err)
		}
	}()

	var redisClient *redisx.Client
	for {
		rc, err := redisx.New(cfg.RedisAddr)
		if err != nil {
			logger.Warn("redis connect failed, retrying", "error", err)
			select {
			case <-ctx.Done():
				os.Exit(1)
			case <-time.After(2 * time.Second):
				continue
			}
		}
		redisClient = rc
		break
	}
	defer redisClient.Close()

	bc := bybit.NewClient(cfg.BybitBaseURL, cfg.BybitAPIKey, cfg.BybitAPISecret)

	var influxWriter *influx.Writer
	if cfg.InfluxToken != "" {
		iw, err := influx.NewWriter(cfg.InfluxURL, cfg.InfluxToken, cfg.InfluxOrg, cfg.InfluxBucketRaw, logger)
		if err != nil {
			logger.Warn("influx writer disabled", "error", err)
		} else {
			influxWriter = iw
			logger.Info("influx trade writer enabled", "url", cfg.InfluxURL, "bucket", cfg.InfluxBucketRaw)
		}
	}

	svc := executor.New(cfg, bc, redisClient, influxWriter, logger)

	logger.Info("oms execution started",
		"bybit_mode", cfg.BybitMode,
		"bybit_base_url", cfg.BybitBaseURL,
		"deposit_usd", cfg.AccountDepositUSD,
		"trade_margin_usd", cfg.TradeMarginUSD,
		"leverage", cfg.Leverage,
		"max_concurrent_trades", risk.MaxConcurrentTrades(cfg.AccountDepositUSD, cfg.TradeMarginUSD),
		"entry_maker_ticks", cfg.EntryMakerTicks,
		"vol_multiplier_cap", cfg.VolMultiplierCap,
		"stale_entry_move_away_pct", cfg.StaleEntryMoveAwayPct,
		"stale_entry_min_age_sec", cfg.StaleEntryMinAgeSec,
		"order_fill_timeout_sec", cfg.OrderFillTimeoutSec,
		"min_sl_pct", cfg.MinSLPct,
		"min_tp_pct", cfg.MinTPPct,
		"max_tp_pct", cfg.MaxTPPct,
		"fee_breakeven_pct", cfg.FeeBreakevenPct,
		"tp_budget_pct", cfg.TPBudgetPct,
		"exit_grid_min_redeploy_sec", cfg.ExitGridMinRedeploySec,
		"min_hold_time_sec", cfg.MinHoldTimeSec,
		"decay_min_hold_sec", cfg.DecayMinHoldSec,
	)
	if err := svc.Run(ctx); err != nil && err != context.Canceled {
		logger.Error("oms exited", "error", err)
		os.Exit(1)
	}
}
