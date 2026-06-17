package redisx

import (
	"context"
	"encoding/json"
	"fmt"
	"time"

	"github.com/redis/go-redis/v9"
	"github.com/vmihailenco/msgpack/v5"
)

type Client struct {
	rdb        *redis.Client
	useMsgPack bool
}

func New(addr string, useMsgPack bool) (*Client, error) {
	rdb := redis.NewClient(&redis.Options{
		Addr:         addr,
		DialTimeout:  5 * time.Second,
		ReadTimeout:  0,
		WriteTimeout: 3 * time.Second,
		PoolSize:     32,
	})
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	if err := rdb.Ping(ctx).Err(); err != nil {
		return nil, fmt.Errorf("redis ping: %w", err)
	}
	return &Client{rdb: rdb, useMsgPack: useMsgPack}, nil
}

func (c *Client) Publish(ctx context.Context, channel string, payload any) error {
	var data []byte
	var err error
	if c.useMsgPack {
		data, err = msgpack.Marshal(payload)
	} else {
		data, err = json.Marshal(payload)
	}
	if err != nil {
		return err
	}
	return c.rdb.Publish(ctx, channel, data).Err()
}

func (c *Client) Subscribe(ctx context.Context, channel string) *redis.PubSub {
	return c.rdb.Subscribe(ctx, channel)
}

func (c *Client) Get(ctx context.Context, key string) (string, error) {
	return c.rdb.Get(ctx, key).Result()
}

func (c *Client) Close() error {
	return c.rdb.Close()
}
