package config

import (
	"os"
	"strconv"
	"time"
)

type Config struct {
	RedisAddr         string
	InfluxURL         string
	InfluxToken       string
	InfluxOrg         string
	InfluxBucket      string
	BatchFlushEvery   time.Duration
	BatchMaxPoints    int
	PointBufferSize   int
	OrderbookDepth    int
	MetricsAddr       string
}

func Load() Config {
	return Config{
		RedisAddr:       envOr("REDIS_ADDR", "redis:6379"),
		InfluxURL:       envOr("INFLUX_URL", "http://influxdb:8086"),
		InfluxToken:     os.Getenv("INFLUX_TOKEN"),
		InfluxOrg:       envOr("INFLUX_ORG", "fasttrader"),
		InfluxBucket:    envOr("INFLUX_BUCKET_RAW", "market_raw"),
		BatchFlushEvery: durationEnv("BATCH_FLUSH_EVERY", 5*time.Second),
		BatchMaxPoints:  intEnv("BATCH_MAX_POINTS", 10000),
		PointBufferSize: intEnv("POINT_BUFFER_SIZE", 50000),
		OrderbookDepth:  intEnv("ORDERBOOK_DEPTH", 50),
		MetricsAddr:     envOr("METRICS_ADDR", ":9104"),
	}
}

func envOr(key, def string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return def
}

func intEnv(key string, def int) int {
	if v := os.Getenv(key); v != "" {
		if i, err := strconv.Atoi(v); err == nil {
			return i
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
