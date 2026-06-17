package main

import (
	"context"
	"encoding/json"
	"log/slog"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/fast-trader-gru/data_ingestion/internal/config"
	"github.com/fast-trader-gru/data_ingestion/internal/metrics"
	"github.com/fast-trader-gru/data_ingestion/internal/models"
	"github.com/fast-trader-gru/data_ingestion/internal/redisx"
	"github.com/fast-trader-gru/data_ingestion/internal/ws"
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
		rc, err := redisx.New(cfg.RedisAddr, cfg.UseMsgPack)
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

	manager := ws.NewManager(cfg, redisClient, logger)

	go subscribeActiveSymbols(ctx, redisClient, cfg, manager, logger)
	go bootstrapRetryLoop(ctx, redisClient, manager, logger)
	go statsLoop(ctx, manager, logger)

	logger.Info("data ingestion started")
	<-ctx.Done()
	manager.Shutdown()
	logger.Info("data ingestion stopped")
}

func bootstrapActiveSymbols(ctx context.Context, redis *redisx.Client, manager *ws.Manager, logger *slog.Logger) bool {
	raw, err := redis.Get(ctx, "config:active_symbols:latest")
	if err != nil {
		return false
	}
	var payload models.ActiveSymbolsPayload
	if err := json.Unmarshal([]byte(raw), &payload); err != nil {
		logger.Warn("invalid cached active symbols", "error", err)
		return false
	}
	if len(payload.Symbols) == 0 {
		return false
	}
	logger.Info("bootstrapped active symbols from cache", "count", len(payload.Symbols))
	manager.UpdateSymbols(payload.Symbols)
	return true
}

// bootstrapRetryLoop re-reads config:active_symbols:latest from Redis every 10s while
// no WebSocket connections are active (startup race with screener pub/sub).
func bootstrapRetryLoop(ctx context.Context, redis *redisx.Client, manager *ws.Manager, logger *slog.Logger) {
	ticker := time.NewTicker(10 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			conns, _, _ := manager.SnapshotStats()
			if conns > 0 {
				continue
			}
			if bootstrapActiveSymbols(ctx, redis, manager, logger) {
				logger.Info("bootstrap retry loaded active symbols from cache")
				continue
			}
			logger.Info("bootstrap retry: no active symbols in cache yet, ws_connections=0")
		}
	}
}

func statsLoop(ctx context.Context, manager *ws.Manager, logger *slog.Logger) {
	ticker := time.NewTicker(60 * time.Second)
	defer ticker.Stop()
	var lastOB, lastTR uint64
	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			conns, ob, tr := manager.SnapshotStats()
			logger.Info("ingestion stats",
				"ws_connections", conns,
				"orderbook_events_1m", ob-lastOB,
				"trade_events_1m", tr-lastTR,
				"orderbook_total", ob,
				"trade_total", tr,
			)
			lastOB, lastTR = ob, tr
		}
	}
}

func subscribeActiveSymbols(ctx context.Context, redis *redisx.Client, cfg config.Config, manager *ws.Manager, logger *slog.Logger) {
	if !bootstrapActiveSymbols(ctx, redis, manager, logger) {
		logger.Info("no cached active symbols yet — will retry from Redis every 10s and listen on pub/sub")
	}
	for {
		if ctx.Err() != nil {
			return
		}
		pubsub := redis.Subscribe(ctx, cfg.ActiveSymbolsChannel)
		ch := pubsub.Channel()
		logger.Info("subscribed to active symbols channel", "channel", cfg.ActiveSymbolsChannel)

		for msg := range ch {
			var payload models.ActiveSymbolsPayload
			if err := json.Unmarshal([]byte(msg.Payload), &payload); err != nil {
				logger.Warn("invalid active symbols payload", "error", err)
				continue
			}
			logger.Info("received active symbols update", "count", len(payload.Symbols))
			manager.UpdateSymbols(payload.Symbols)
		}

		if ctx.Err() != nil {
			pubsub.Close()
			return
		}
		logger.Warn("redis subscription dropped, reconnecting")
		pubsub.Close()
		time.Sleep(2 * time.Second)
	}
}
