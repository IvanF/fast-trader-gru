package ws

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/fast-trader-gru/data_ingestion/internal/config"
	"github.com/fast-trader-gru/data_ingestion/internal/metrics"
	"github.com/fast-trader-gru/data_ingestion/internal/models"
	"github.com/fast-trader-gru/data_ingestion/internal/redisx"
	"github.com/gorilla/websocket"
)

type Manager struct {
	cfg    config.Config
	redis  *redisx.Client
	logger *slog.Logger

	mu       sync.RWMutex
	symbols  map[string]struct{}
	conns    map[string]*symbolConn
	cancelFn context.CancelFunc

	obEvents uint64
	trEvents uint64
}

type symbolConn struct {
	cancel context.CancelFunc
}

func NewManager(cfg config.Config, redis *redisx.Client, logger *slog.Logger) *Manager {
	return &Manager{
		cfg:     cfg,
		redis:   redis,
		logger:  logger,
		symbols: make(map[string]struct{}),
		conns:   make(map[string]*symbolConn),
	}
}

func (m *Manager) UpdateSymbols(symbols []string) {
	m.mu.Lock()
	defer m.mu.Unlock()

	desired := make(map[string]struct{}, len(symbols))
	for _, s := range symbols {
		desired[s] = struct{}{}
	}

	for sym := range m.symbols {
		if _, ok := desired[sym]; !ok {
			if c, exists := m.conns[sym]; exists {
				c.cancel()
				delete(m.conns, sym)
			}
			delete(m.symbols, sym)
			m.logger.Info("closed ws for symbol", "symbol", sym)
		}
	}

	for sym := range desired {
		if _, ok := m.symbols[sym]; !ok {
			ctx, cancel := context.WithCancel(context.Background())
			m.conns[sym] = &symbolConn{cancel: cancel}
			m.symbols[sym] = struct{}{}
			go m.runSymbolWS(ctx, sym)
			m.logger.Info("opened ws for symbol", "symbol", sym)
		}
	}
	metrics.ActiveConnections.Set(float64(len(m.conns)))
}

func (m *Manager) runSymbolWS(ctx context.Context, symbol string) {
	backoff := m.cfg.ReconnectBaseDelay
	for {
		if ctx.Err() != nil {
			return
		}
		err := m.connectAndStream(ctx, symbol)
		if ctx.Err() != nil {
			return
		}
		m.logger.Warn("ws disconnected, reconnecting", "symbol", symbol, "error", err)
		metrics.ReconnectTotal.Inc()
		time.Sleep(backoff)
		if backoff < m.cfg.MaxReconnectDelay {
			backoff *= 2
			if backoff > m.cfg.MaxReconnectDelay {
				backoff = m.cfg.MaxReconnectDelay
			}
		}
	}
}

func (m *Manager) connectAndStream(ctx context.Context, symbol string) error {
	conn, _, err := websocket.DefaultDialer.DialContext(ctx, m.cfg.BybitWSURL, nil)
	if err != nil {
		return fmt.Errorf("dial: %w", err)
	}
	defer conn.Close()

	sub := map[string]any{
		"op": "subscribe",
		"args": []string{
			fmt.Sprintf("orderbook.50.%s", symbol),
			fmt.Sprintf("publicTrade.%s", symbol),
		},
	}
	if err := conn.WriteJSON(sub); err != nil {
		return fmt.Errorf("subscribe: %w", err)
	}

	conn.SetReadDeadline(time.Now().Add(60 * time.Second))
	conn.SetPongHandler(func(string) error {
		conn.SetReadDeadline(time.Now().Add(60 * time.Second))
		return nil
	})

	pingTicker := time.NewTicker(20 * time.Second)
	defer pingTicker.Stop()

	errCh := make(chan error, 1)
	go func() {
		for {
			_, msg, err := conn.ReadMessage()
			if err != nil {
				errCh <- err
				return
			}
			conn.SetReadDeadline(time.Now().Add(60 * time.Second))
			if err := m.handleMessage(ctx, symbol, msg); err != nil {
				m.logger.Warn("message handling error", "symbol", symbol, "error", err)
			}
		}
	}()

	for {
		select {
		case <-ctx.Done():
			return ctx.Err()
		case err := <-errCh:
			return err
		case <-pingTicker.C:
			if err := conn.WriteControl(websocket.PingMessage, []byte("ping"), time.Now().Add(5*time.Second)); err != nil {
				return err
			}
		}
	}
}

func (m *Manager) handleMessage(ctx context.Context, symbol string, raw []byte) error {
	var envelope struct {
		Topic string          `json:"topic"`
		Type  string          `json:"type"`
		Ts    int64           `json:"ts"`
		Data  json.RawMessage `json:"data"`
	}
	if err := json.Unmarshal(raw, &envelope); err != nil {
		return err
	}
	if envelope.Topic == "" {
		return nil
	}

	now := time.Now()
	if strings.HasPrefix(envelope.Topic, "orderbook.") {
		atomic.AddUint64(&m.obEvents, 1)
		metrics.WSEventsTotal.WithLabelValues("orderbook").Inc()
		var data struct {
			S  string              `json:"s"`
			B  []models.OrderbookLevel `json:"b"`
			A  []models.OrderbookLevel `json:"a"`
			U  uint64              `json:"u"`
			Seq uint64             `json:"seq"`
		}
		if err := json.Unmarshal(envelope.Data, &data); err != nil {
			return err
		}
		payload := models.OrderbookPayload{
			Symbol:   symbol,
			Ts:       envelope.Ts,
			Bids:     data.B,
			Asks:     data.A,
			UpdateID: data.U,
			Seq:      data.Seq,
		}
		ch := fmt.Sprintf("market:orderbook:%s", symbol)
		if err := m.redis.Publish(ctx, ch, payload); err != nil {
			return err
		}
		metrics.RedisPublishTotal.WithLabelValues("orderbook").Inc()
		metrics.PublishLatency.Observe(time.Since(now).Seconds())
		return nil
	}

	if strings.HasPrefix(envelope.Topic, "publicTrade.") {
		atomic.AddUint64(&m.trEvents, 1)
		metrics.WSEventsTotal.WithLabelValues("trade").Inc()
		var trades []struct {
			T  models.FlexInt64 `json:"T"`
			S  string           `json:"S"`
			V  string           `json:"v"`
			P  string           `json:"p"`
			I  string           `json:"i"`
			BT bool             `json:"BT"`
		}
		if err := json.Unmarshal(envelope.Data, &trades); err != nil {
			return err
		}
		for _, t := range trades {
			price, _ := strconv.ParseFloat(t.P, 64)
			size, _ := strconv.ParseFloat(t.V, 64)
			payload := models.TradePayload{
				Symbol:  symbol,
				Ts:      int64(t.T),
				Price:   price,
				Size:    size,
				Side:    t.S,
				TradeID: t.I,
				IsBlock: t.BT,
			}
			ch := fmt.Sprintf("market:trades:%s", symbol)
			if err := m.redis.Publish(ctx, ch, payload); err != nil {
				return err
			}
			metrics.RedisPublishTotal.WithLabelValues("trade").Inc()
		}
		metrics.PublishLatency.Observe(time.Since(now).Seconds())
	}
	return nil
}

func (m *Manager) Shutdown() {
	m.mu.Lock()
	defer m.mu.Unlock()
	for sym, c := range m.conns {
		c.cancel()
		delete(m.conns, sym)
		delete(m.symbols, sym)
		m.logger.Info("shutdown ws", "symbol", sym)
	}
	metrics.ActiveConnections.Set(0)
}

func (m *Manager) SnapshotStats() (activeConns int, orderbookEvents, tradeEvents uint64) {
	m.mu.RLock()
	activeConns = len(m.conns)
	m.mu.RUnlock()
	return activeConns, atomic.LoadUint64(&m.obEvents), atomic.LoadUint64(&m.trEvents)
}
