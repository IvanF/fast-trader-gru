package config

import (
	"os"
	"time"
)

type Config struct {
	RedisAddr            string
	BybitWSURL           string
	ActiveSymbolsChannel string
	MetricsAddr          string
	ReconnectBaseDelay   time.Duration
	MaxReconnectDelay    time.Duration
	UseMsgPack           bool
}

func Load() Config {
	return Config{
		RedisAddr:            envOr("REDIS_ADDR", "redis:6379"),
		BybitWSURL:           envOr("BYBIT_WS_URL", "wss://stream.bybit.com/v5/public/linear"),
		ActiveSymbolsChannel: envOr("ACTIVE_SYMBOLS_CHANNEL", "config:active_symbols"),
		MetricsAddr:          envOr("METRICS_ADDR", ":9101"),
		ReconnectBaseDelay:   500 * time.Millisecond,
		MaxReconnectDelay:    30 * time.Second,
		UseMsgPack:           envOr("USE_MSGPACK", "true") == "true",
	}
}

func envOr(key, def string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return def
}
