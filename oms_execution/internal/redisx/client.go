package redisx

import (
	"context"
	"encoding/json"
	"fmt"
	"time"

	"github.com/fast-trader-gru/oms_execution/internal/models"
	"github.com/redis/go-redis/v9"
)

type Client struct {
	rdb *redis.Client
}

func New(addr string) (*Client, error) {
	rdb := redis.NewClient(&redis.Options{
		Addr:         addr,
		DialTimeout:  5 * time.Second,
		ReadTimeout:  0,
		WriteTimeout: 3 * time.Second,
		PoolSize:     16,
	})
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	if err := rdb.Ping(ctx).Err(); err != nil {
		return nil, fmt.Errorf("redis ping: %w", err)
	}
	return &Client{rdb: rdb}, nil
}

func (c *Client) Subscribe(ctx context.Context, channel string) *redis.PubSub {
	return c.rdb.Subscribe(ctx, channel)
}

func (c *Client) Publish(ctx context.Context, channel string, payload any) error {
	data, err := json.Marshal(payload)
	if err != nil {
		return err
	}
	return c.rdb.Publish(ctx, channel, data).Err()
}

func (c *Client) GetOrderbook(ctx context.Context, symbol string) (models.OrderbookSnapshot, error) {
	key := fmt.Sprintf("cache:orderbook:%s", symbol)
	val, err := c.rdb.Get(ctx, key).Result()
	if err == redis.Nil {
		return models.OrderbookSnapshot{}, fmt.Errorf("no cached orderbook for %s", symbol)
	}
	if err != nil {
		return models.OrderbookSnapshot{}, err
	}
	var ob models.OrderbookSnapshot
	if err := json.Unmarshal([]byte(val), &ob); err != nil {
		return models.OrderbookSnapshot{}, err
	}
	return ob, nil
}

func (c *Client) SetOrderbook(ctx context.Context, symbol string, ob models.OrderbookSnapshot) error {
	key := fmt.Sprintf("cache:orderbook:%s", symbol)
	data, err := json.Marshal(ob)
	if err != nil {
		return err
	}
	return c.rdb.Set(ctx, key, data, 30*time.Second).Err()
}

func (c *Client) PSubscribe(ctx context.Context, patterns ...string) *redis.PubSub {
	return c.rdb.PSubscribe(ctx, patterns...)
}

func (c *Client) Close() error {
	return c.rdb.Close()
}

func (c *Client) GetActiveSymbols(ctx context.Context) ([]string, error) {
	val, err := c.rdb.Get(ctx, "config:active_symbols:latest").Result()
	if err == redis.Nil {
		return nil, nil
	}
	if err != nil {
		return nil, err
	}
	var payload struct {
		Symbols []string `json:"symbols"`
	}
	if err := json.Unmarshal([]byte(val), &payload); err != nil {
		return nil, err
	}
	return payload.Symbols, nil
}

// --- Position persistence ---

func posKey(symbol string) string {
	return "oms:position:" + symbol
}

func (c *Client) SavePosition(ctx context.Context, pos *models.ActivePosition) error {
	data, err := json.Marshal(pos)
	if err != nil {
		return err
	}
	pipe := c.rdb.Pipeline()
	pipe.Set(ctx, posKey(pos.Symbol), data, 0)
	pipe.SAdd(ctx, "oms:positions", pos.Symbol)
	_, err = pipe.Exec(ctx)
	return err
}

func (c *Client) DeletePosition(ctx context.Context, symbol string) error {
	pipe := c.rdb.Pipeline()
	pipe.Del(ctx, posKey(symbol))
	pipe.SRem(ctx, "oms:positions", symbol)
	_, err := pipe.Exec(ctx)
	return err
}

func (c *Client) LoadPositions(ctx context.Context) ([]*models.ActivePosition, error) {
	symbols, err := c.rdb.SMembers(ctx, "oms:positions").Result()
	if err != nil {
		return nil, err
	}
	var positions []*models.ActivePosition
	for _, sym := range symbols {
		val, err := c.rdb.Get(ctx, posKey(sym)).Result()
		if err == redis.Nil {
			c.rdb.SRem(context.Background(), "oms:positions", sym)
			continue
		}
		if err != nil {
			continue
		}
		var pos models.ActivePosition
		if err := json.Unmarshal([]byte(val), &pos); err != nil {
			continue
		}
		positions = append(positions, &pos)
	}
	return positions, nil
}
