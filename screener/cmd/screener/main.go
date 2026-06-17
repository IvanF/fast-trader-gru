package main

import (
	"context"
	"log/slog"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/fast-trader-gru/screener/internal/bybit"
	"github.com/fast-trader-gru/screener/internal/config"
	"github.com/fast-trader-gru/screener/internal/metrics"
	"github.com/fast-trader-gru/screener/internal/redisx"
	"github.com/fast-trader-gru/screener/internal/screener"
)

func main() {
	logger := slog.New(slog.NewJSONHandler(os.Stdout, &slog.HandlerOptions{Level: slog.LevelInfo}))
	cfg := config.Load()

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

	svc := screener.New(cfg, bybit.NewClient(cfg.BybitBaseURL), redisClient, logger)
	logger.Info("screener started", "interval", cfg.ScreenInterval.String())
	if err := svc.Run(ctx); err != nil && err != context.Canceled {
		logger.Error("screener exited", "error", err)
		os.Exit(1)
	}
}
