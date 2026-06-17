package config

import (
	"os"
	"strconv"
	"strings"
	"time"
)

type Config struct {
	RedisAddr            string
	BybitBaseURL         string
	ScreenInterval       time.Duration
	MinTurnover24h       float64
	MaxFundingRate       float64
	MinFundingRate       float64
	MetricsAddr          string
	ActiveSymbolsChannel string
	BlacklistSymbols     map[string]struct{}
}

func Load() Config {
	return Config{
		RedisAddr:            envOr("REDIS_ADDR", "redis:6379"),
		BybitBaseURL:         envOr("BYBIT_BASE_URL", "https://api.bybit.com"),
		ScreenInterval:       durationEnv("SCREEN_INTERVAL", time.Hour),
		MinTurnover24h:       floatEnv("MIN_TURNOVER_24H", 50_000_000),
		MaxFundingRate:       floatEnv("MAX_FUNDING_RATE", 0.001),
		MinFundingRate:       floatEnv("MIN_FUNDING_RATE", -0.001),
		MetricsAddr:          envOr("METRICS_ADDR", ":9100"),
		ActiveSymbolsChannel: envOr("ACTIVE_SYMBOLS_CHANNEL", "config:active_symbols"),
		BlacklistSymbols:     symbolSetEnv("BLACKLIST_SYMBOLS"),
	}
}

func symbolSetEnv(key string) map[string]struct{} {
	raw := os.Getenv(key)
	if raw == "" {
		return nil
	}
	out := make(map[string]struct{})
	for _, part := range strings.Split(raw, ",") {
		sym := strings.ToUpper(strings.TrimSpace(part))
		if sym != "" {
			out[sym] = struct{}{}
		}
	}
	return out
}

func envOr(key, def string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return def
}

func floatEnv(key string, def float64) float64 {
	if v := os.Getenv(key); v != "" {
		if f, err := strconv.ParseFloat(v, 64); err == nil {
			return f
		}
	}
	return def
}

func durationEnv(key string, def time.Duration) time.Duration {
	if v := os.Getenv(key); v != "" {
		if d, err := time.ParseDuration(v); err == nil {
			return d
		}
	}
	return def
}
