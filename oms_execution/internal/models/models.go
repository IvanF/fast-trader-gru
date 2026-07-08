package models

type TradeSignal struct {
	Symbol               string    `json:"symbol"`
	Direction            string    `json:"direction"`
	Confidence           float64   `json:"confidence"`
	VolatilityMultiplier float64   `json:"volatility_multiplier"`
	StateVector          []float32 `json:"state_vector"`
	FundingRate          float64   `json:"funding_rate"`
	Regime               string    `json:"regime"`
	BTCorrelation        float64   `json:"btc_correlation"`
	PositionScale        float64   `json:"position_scale"`
	EntryPrice           float64   `json:"entry_price"`
	StopLoss             float64   `json:"stop_loss"`
	TakeProfits          []float64 `json:"take_profits"`
	TPPrices             []float64 `json:"tp_prices,omitempty"`
	WallPrice            float64   `json:"wall_price"`
	Timestamp            int64     `json:"timestamp"`
	SignalID             string    `json:"signal_id"`
	ExitReason           string    `json:"exit_reason,omitempty"`
	SetupAction          string    `json:"setup_action,omitempty"`
	AbortReason          string    `json:"abort_reason,omitempty"`
	DecayReason          string    `json:"decay_reason,omitempty"`
	MacroTrend5m         float64   `json:"macro_trend_5m,omitempty"`
	MacroTrend15m        float64   `json:"macro_trend_15m,omitempty"`
	DynamicSLPct         float64   `json:"dynamic_sl_pct,omitempty"`
	DynamicTPPct         float64   `json:"dynamic_tp_pct,omitempty"`
	ShadowOnly           bool      `json:"shadow_only,omitempty"`
	ATRPct               float64   `json:"atr_14_pct,omitempty"` // Normalized ATR for dynamic mode detection
}

// PendingOrderEvent notifies ML of pending entry lifecycle for alpha-decay abort.
type PendingOrderEvent struct {
	Event      string  `json:"event"`
	Symbol     string  `json:"symbol"`
	Direction  string  `json:"direction"`
	OrderID    string  `json:"order_id"`
	EntryPrice float64 `json:"entry_price"`
	Confidence float64 `json:"confidence"`
	SignalID   string  `json:"signal_id"`
	Reason     string  `json:"reason,omitempty"`
	Timestamp  int64   `json:"timestamp"`
}

// PositionEvent notifies ML of lifecycle changes for alpha-decay tracking.
type PositionEvent struct {
	Event      string  `json:"event"`
	Symbol     string  `json:"symbol"`
	Direction  string  `json:"direction"`
	SignalID   string  `json:"signal_id"`
	Confidence float64 `json:"confidence"`
	EntryPrice float64 `json:"entry_price"`
	Timestamp  int64   `json:"timestamp"`
}

type OrderbookLevel struct {
	Price string `json:"price" msgpack:"price"`
	Size  string `json:"size" msgpack:"size"`
}

type GKSnapshot struct {
	SpreadPct     float64
	OBI            float64
	Momentum       float64
	PriceVelocity  float64
	ATRPct         float64
	VolumeRatio    float64
}

type OrderbookSnapshot struct {
	Symbol string           `json:"symbol" msgpack:"symbol"`
	Ts     int64            `json:"ts" msgpack:"ts"`
	Bids   []OrderbookLevel `json:"b" msgpack:"b"`
	Asks   []OrderbookLevel `json:"a" msgpack:"a"`
}

type ExecutionResult struct {
	SignalID      string    `json:"signal_id"`
	Symbol        string    `json:"symbol"`
	Direction     string    `json:"direction"`
	StateVector   []float32 `json:"state_vector"`
	FAISSVectorID int64     `json:"faiss_vector_id"`
	EntryPrice    float64   `json:"entry_price"`
	ExitPrice     float64   `json:"exit_price"`
	NetPnL        float64   `json:"net_pnl"`
	HoldingTimeMs int64     `json:"holding_time_ms"`
	Regime        string    `json:"regime"`
	ClosedAt      int64     `json:"closed_at"`
	PartialClosed bool      `json:"partial_closed"`
	GridLevels    int       `json:"grid_levels"`
	CloseReason   string    `json:"close_reason"`
	ExchangePnL   bool      `json:"exchange_pnl"`
}

type GridPlan struct {
	Symbol       string
	Direction    string
	EntryPrice   float64
	StopLoss     float64
	TakeProfits  []float64
	Qty          float64
	TimeStopSec  int
	Signal       TradeSignal
	WallPrice    float64
}

type ExitOrder struct {
	OrderID   string
	Price     float64
	Qty       float64
	Kind      string // stop_loss | breakeven | wall | r_multiple | trend | time_stop | signal_exit | confidence_decay_exit
	Filled    bool
	FilledQty float64
	FilledPx  float64
	IsStop    bool // true for conditional stop-market orders
}

type ActivePosition struct {
	Symbol          string
	Direction       string
	FillPrice       float64
	PlannedEntry    float64
	PlannedSL       float64
	InitialQty      float64
	TargetQty       float64
	RemainingQty    float64
	StopLoss        float64
	StopLossOrder   *ExitOrder
	TakeProfitOrders []ExitOrder
	PartialTaken    bool
	BreakevenSet    bool
	EntryTime       int64
	TimeStopSec     int
	QtyStep         float64
	MinOrderQty     float64
	TickSize        float64
	MarginUSD       float64
	NotionalUSD     float64
	Leverage        int
	Signal          TradeSignal
	OrderID         string
	FilledAt        int64
	ExitGridReady    bool
	TimeStopPlaced   bool
	LastGridDeployAt int64
	EmergencySizeHandled  bool
	GridDeployFailures    int
	LastGridDeployFailure int64
	// PositionManager fields
	ScaledOut       bool      // Scale-out at 1R executed?
	BreakevenPMSet  bool      // Breakeven at 1.5R executed?
	OriginalRisk    float64   // Original SL distance from entry
	PriceHistory    []float64 // Rolling mid prices for ATR
	EntryCandleIdx  int       // Candle index at entry time
	CandleHigh      float64   // Current candle high
	CandleLow       float64   // Current candle low
	// Dynamic Trading Mode (0=Normal, 1=HFT Scalping)
	TradingMode     int       // 0=Normal, 1=HFTScalping — set by DetectTradingMode at entry
	// Gatekeeper entry-time feature snapshot (captured at entry for InfluxDB logging)
	SpreadPctAtEntry      float64
	OBIAtEntry            float64
	MomentumAtEntry       float64
	PriceVelocityAtEntry  float64
	ATRPctAtEntry         float64
	VolumeRatioAtEntry    float64
	OpenPositionsAtEntry  int
	// MFE/MAE tracking for smart labeling
	MaxFavorablePrice float64
	MaxAdversePrice   float64
}

type PendingEntryState string

const (
	PendingEntryStateActive     PendingEntryState = "active"
	PendingEntryStateCancelling PendingEntryState = "cancelling"
)

type PendingEntry struct {
	Symbol       string
	OrderID      string
	State        PendingEntryState
	Direction    string
	EntryPrice   float64
	StopLoss     float64
	TakeProfits  []float64
	Qty          float64
	TimeStopSec  int
	QtyStep      float64
	MinOrderQty  float64
	TickSize     float64
	MarginUSD    float64
	NotionalUSD  float64
	Leverage     int
	Signal       TradeSignal
	PlacedAt     int64
	Orderbook    OrderbookSnapshot
	// Gatekeeper entry-time feature snapshot
	GKSpreadPct     float64
	GKOBI           float64
	GKMomentum      float64
	GKPriceVelocity float64
	GKATRPct        float64
	GKVolumeRatio   float64
}
