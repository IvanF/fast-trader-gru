package config

import (
	"os"
	"strconv"
)

type Config struct {
	RedisAddr            string
	BybitBaseURL         string
	BybitMode            string
	BybitAPIKey          string
	BybitAPISecret       string
	SignalsChannel       string
	ResultsChannel       string
	PositionsChannel     string
	PendingOrdersChannel string
	MetricsAddr          string
	AccountDepositUSD    float64
	TradeMarginUSD       float64
	Leverage             int
	DefaultQty           float64
	TimeStopSeconds      int
	PartialClosePct      float64
	Demo                 bool
	Testnet              bool
	OrderFillTimeoutSec  int
	InfluxURL            string
	InfluxToken          string
	InfluxOrg            string
	InfluxBucketRaw      string
	ConfidenceThreshold       float64
	KillConfidenceThreshold   float64
	EntryRepriceThresholdPct   float64
	ExitRepriceThresholdPct    float64
	StaleEntryMoveAwayPct      float64
	StaleEntryMinAgeSec        int
	EntryMakerTicks            int
	VolMultiplierCap           float64
	PendingVolRepriceDelta     float64
	MinSLPct                   float64
	MaxSLPct                   float64
	MinTPPct                   float64
	MaxTPPct                   float64
	FeeBreakevenPct            float64
	TPBudgetPct                float64
	ExitGridMinRedeploySec     int
	MinHoldTimeSec             int
	DecayMinHoldSec            int
	EntryFeeRate               float64
	ExitFeeRate                float64
	TargetNetProfitPct         float64
	SymbolOverrides            map[string]SymbolConfig
	ShadowMode                 bool
	ShadowAlwaysEnabled        bool
	ShadowTimeStopSec          int
}

type SymbolConfig struct {
	MinSLPct        *float64
	MaxSLPct        *float64
	TradeMarginUSD  *float64
	Leverage        *int
	TimeStopSeconds *int
}

func (c *Config) GetMinSLPct(symbol string) float64 {
	if ov, ok := c.SymbolOverrides[symbol]; ok && ov.MinSLPct != nil {
		return *ov.MinSLPct
	}
	return c.MinSLPct
}

func (c *Config) GetTradeMarginUSD(symbol string) float64 {
	if ov, ok := c.SymbolOverrides[symbol]; ok && ov.TradeMarginUSD != nil {
		return *ov.TradeMarginUSD
	}
	return c.TradeMarginUSD
}

func (c *Config) GetMaxSLPct(symbol string) float64 {
	if ov, ok := c.SymbolOverrides[symbol]; ok && ov.MaxSLPct != nil {
		return *ov.MaxSLPct
	}
	return c.MaxSLPct
}

func (c *Config) GetLeverage(symbol string) int {
	if ov, ok := c.SymbolOverrides[symbol]; ok && ov.Leverage != nil {
		return *ov.Leverage
	}
	return c.Leverage
}

func (c *Config) GetTimeStopSeconds(symbol string) int {
	if ov, ok := c.SymbolOverrides[symbol]; ok && ov.TimeStopSeconds != nil {
		return *ov.TimeStopSeconds
	}
	return c.TimeStopSeconds
}

func Load() Config {
	demo := envOr("BYBIT_DEMO", "false") == "true"
	testnet := envOr("BYBIT_TESTNET", "false") == "true"
	mode, baseURL := ResolveBybitREST(demo, testnet, os.Getenv("BYBIT_BASE_URL"))
	tradeMargin := floatEnv("TRADE_MARGIN_USD", 10)
	return Config{
		RedisAddr:       envOr("REDIS_ADDR", "redis:6379"),
		BybitBaseURL:    baseURL,
		BybitMode:       mode,
		BybitAPIKey:     os.Getenv("BYBIT_API_KEY"),
		BybitAPISecret:  os.Getenv("BYBIT_API_SECRET"),
		SignalsChannel:    envOr("SIGNALS_CHANNEL", "orders:signals"),
		ResultsChannel:    envOr("RESULTS_CHANNEL", "execution:results"),
		PositionsChannel:     envOr("POSITIONS_CHANNEL", "execution:positions"),
		PendingOrdersChannel: envOr("PENDING_ORDERS_CHANNEL", "execution:pending_orders"),
		MetricsAddr:     envOr("METRICS_ADDR", ":9102"),
		AccountDepositUSD: floatEnv("ACCOUNT_DEPOSIT_USD", 100),
		TradeMarginUSD:    tradeMargin,
		Leverage:          intEnv("LEVERAGE", 5),
		DefaultQty:        floatEnv("DEFAULT_ORDER_QTY", 0.01),
		TimeStopSeconds:   intEnv("TIME_STOP_SECONDS", 3600),
		PartialClosePct:   floatEnv("PARTIAL_CLOSE_PCT", 0.45),
		Demo:              demo,
		Testnet:           testnet,
		OrderFillTimeoutSec: intEnv("ORDER_FILL_TIMEOUT_SEC", 3600),
		InfluxURL:         envOr("INFLUX_URL", "http://influxdb:8086"),
		InfluxToken:       os.Getenv("INFLUX_TOKEN"),
		InfluxOrg:         envOr("INFLUX_ORG", "fasttrader"),
		InfluxBucketRaw:   envOr("INFLUX_BUCKET_RAW", "market_raw"),
		ConfidenceThreshold:        floatEnv("CONFIDENCE_THRESHOLD", 0.65),
		KillConfidenceThreshold:    floatEnv("KILL_CONFIDENCE_THRESHOLD", 0.35),
		EntryRepriceThresholdPct:   floatEnv("ENTRY_REPRICE_THRESHOLD_PCT", 0.002),
		ExitRepriceThresholdPct:    floatEnv("EXIT_REPRICE_THRESHOLD_PCT", 0.003),
		StaleEntryMoveAwayPct:      floatEnv("STALE_ENTRY_MOVE_AWAY_PCT", 0.008),
		StaleEntryMinAgeSec:        intEnv("STALE_ENTRY_MIN_AGE_SEC", 3),
		EntryMakerTicks:            intEnv("ENTRY_MAKER_TICKS", 2),
		VolMultiplierCap:           floatEnv("VOL_MULTIPLIER_CAP", 2.0),
		PendingVolRepriceDelta:     floatEnv("PENDING_VOL_REPRICE_DELTA", 0.5),
		MinSLPct:                   floatEnv("MIN_SL_PCT", 0.003),
		MaxSLPct:                   floatEnv("MAX_SL_PCT", 0.012),
		MinTPPct:                   floatEnv("MIN_TP_PCT", 0.002),
		MaxTPPct:                   floatEnv("MAX_TP_PCT", 0.008),
		FeeBreakevenPct:            floatEnv("FEE_BREAKEVEN_PCT", 0.0015),
		TPBudgetPct:                floatEnv("TP_BUDGET_PCT", 0.35),
		ExitGridMinRedeploySec:     intEnv("EXIT_GRID_MIN_REDEPLOY_SEC", 15),
		MinHoldTimeSec:             intEnv("MIN_HOLD_TIME_SEC", 0),
		DecayMinHoldSec:            intEnv("DECAY_MIN_HOLD_SEC", 0),
		EntryFeeRate:               floatEnv("ENTRY_FEE_RATE", 0.00055),
		ExitFeeRate:                floatEnv("EXIT_FEE_RATE", 0.0002),
		TargetNetProfitPct:         floatEnv("TARGET_NET_PROFIT_PCT", 0.002),
		SymbolOverrides:            parseSymbolOverrides(),
		ShadowMode:                 envOr("SHADOW_MODE", "false") == "true",
		ShadowAlwaysEnabled:        envOr("SHADOW_ALWAYS_ENABLED", "false") == "true",
		ShadowTimeStopSec:          intEnv("SHADOW_TIME_STOP_SEC", 1800),
	}
}

func parseSymbolOverrides() map[string]SymbolConfig {
	overrides := make(map[string]SymbolConfig)
	symbols := []string{"LAB", "BTC", "ETH", "SOL", "XRP", "NEAR", "HYPE", "SPCX", "WLD", "ZEC", "BTW", "UB"}
	for _, sym := range symbols {
		var sc SymbolConfig
		slKey := sym + "_MIN_SL_PCT"
		maxSlKey := sym + "_MAX_SL_PCT"
		marginKey := sym + "_TRADE_MARGIN_USD"
		levKey := sym + "_LEVERAGE"
		timeStopKey := sym + "_TIME_STOP_SECONDS"
		if v := os.Getenv(slKey); v != "" {
			if f, err := strconv.ParseFloat(v, 64); err == nil {
				sc.MinSLPct = &f
			}
		}
		if v := os.Getenv(maxSlKey); v != "" {
			if f, err := strconv.ParseFloat(v, 64); err == nil {
				sc.MaxSLPct = &f
			}
		}
		if v := os.Getenv(marginKey); v != "" {
			if f, err := strconv.ParseFloat(v, 64); err == nil {
				sc.TradeMarginUSD = &f
			}
		}
		if v := os.Getenv(levKey); v != "" {
			if i, err := strconv.Atoi(v); err == nil {
				sc.Leverage = &i
			}
		}
		if v := os.Getenv(timeStopKey); v != "" {
			if i, err := strconv.Atoi(v); err == nil {
				sc.TimeStopSeconds = &i
			}
		}
		if sc.MinSLPct != nil || sc.MaxSLPct != nil || sc.TradeMarginUSD != nil || sc.Leverage != nil || sc.TimeStopSeconds != nil {
			overrides[sym+"USDT"] = sc
		}
	}
	return overrides
}

func (c Config) UsesUSDSizing() bool {
	return c.TradeMarginUSD > 0 && c.Leverage > 0
}

// ResolveBybitREST picks the REST base URL for signed trading endpoints.
// Demo and testnet are mutually exclusive per Bybit docs.
func ResolveBybitREST(demo, testnet bool, explicitURL string) (mode, url string) {
	if demo {
		return "demo", "https://api-demo.bybit.com"
	}
	if testnet {
		return "testnet", "https://api-testnet.bybit.com"
	}
	if explicitURL != "" {
		return "custom", explicitURL
	}
	return "mainnet", "https://api.bybit.com"
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

func intEnv(key string, def int) int {
	if v := os.Getenv(key); v != "" {
		if i, err := strconv.Atoi(v); err == nil {
			return i
		}
	}
	return def
}
