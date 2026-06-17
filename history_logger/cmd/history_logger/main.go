package main

import (
	"context"
	"log/slog"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/fast-trader-gru/history_logger/internal/config"
	"github.com/fast-trader-gru/history_logger/internal/influx"
	"github.com/fast-trader-gru/history_logger/internal/metrics"
	"github.com/fast-trader-gru/history_logger/internal/subscriber"
	"github.com/redis/go-redis/v9"
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

	var rdb *redis.Client
	for {
		rdb = redis.NewClient(&redis.Options{
			Addr:         cfg.RedisAddr,
			DialTimeout:  5 * time.Second,
			ReadTimeout:  0,
			WriteTimeout: 3 * time.Second,
			PoolSize:     8,
		})
		if err := rdb.Ping(ctx).Err(); err != nil {
			logger.Warn("redis connect failed, retrying", "error", err)
			select {
			case <-ctx.Done():
				os.Exit(1)
			case <-time.After(2 * time.Second):
				continue
			}
		}
		break
	}
	defer rdb.Close()

	for {
		if err := influx.HealthCheck(ctx, cfg); err != nil {
			logger.Warn("influxdb not ready, retrying", "error", err)
			select {
			case <-ctx.Done():
				os.Exit(1)
			case <-time.After(2 * time.Second):
				continue
			}
		}
		break
	}

	writer, err := influx.NewBatchWriter(cfg, logger)
	if err != nil {
		logger.Error("failed to create batch writer", "error", err)
		os.Exit(1)
	}
	defer writer.Close()

	svc := subscriber.New(cfg, rdb, writer, logger)
	logger.Info("history_logger started",
		"bucket", cfg.InfluxBucket,
		"flush_every", cfg.BatchFlushEvery.String(),
		"max_batch", cfg.BatchMaxPoints,
	)
	if err := svc.Run(ctx); err != nil && err != context.Canceled {
		logger.Error("history_logger exited", "error", err)
		os.Exit(1)
	}
}
