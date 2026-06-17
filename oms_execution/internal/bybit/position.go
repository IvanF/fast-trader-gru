package bybit

import (
	"context"
	"encoding/json"
	"fmt"
	"net/url"
	"strconv"
	"time"
)

type PositionInfo struct {
	Symbol   string
	Side     string // Buy | Sell
	Size     float64
	AvgPrice float64
}

type OrderInfo struct {
	OrderID     string
	OrderStatus string // New, PartiallyFilled, Filled, Cancelled, Rejected
	AvgPrice    float64
	CumExecQty  float64
	Price       float64
	Side        string
	Qty         float64
	ReduceOnly  bool
}

type ClosedPnLRecord struct {
	Symbol        string
	ClosedPnL     float64
	AvgEntryPrice float64
	AvgExitPrice  float64
	Qty           float64
	UpdatedTime   int64
}

func (c *Client) GetPosition(ctx context.Context, symbol string) (PositionInfo, error) {
	q := url.Values{
		"category": {"linear"},
		"symbol":   {symbol},
	}
	var resp APIResponse
	if err := c.signedGet(ctx, "/v5/position/list", q, &resp); err != nil {
		return PositionInfo{}, err
	}
	if resp.RetCode != 0 {
		return PositionInfo{}, fmt.Errorf("position list: %s (code %d)", resp.RetMsg, resp.RetCode)
	}
	var result struct {
		List []struct {
			Symbol   string `json:"symbol"`
			Side     string `json:"side"`
			Size     string `json:"size"`
			AvgPrice string `json:"avgPrice"`
		} `json:"list"`
	}
	if err := json.Unmarshal(resp.Result, &result); err != nil {
		return PositionInfo{}, err
	}
	if len(result.List) == 0 {
		return PositionInfo{Symbol: symbol}, nil
	}
	item := result.List[0]
	return PositionInfo{
		Symbol:   item.Symbol,
		Side:     item.Side,
		Size:     parseFloatOr(item.Size, 0),
		AvgPrice: parseFloatOr(item.AvgPrice, 0),
	}, nil
}

// ListOpenPositions returns all non-zero linear USDT-margined positions.
func (c *Client) ListOpenPositions(ctx context.Context) ([]PositionInfo, error) {
	q := url.Values{
		"category":   {"linear"},
		"settleCoin": {"USDT"},
	}
	var resp APIResponse
	if err := c.signedGet(ctx, "/v5/position/list", q, &resp); err != nil {
		return nil, err
	}
	if resp.RetCode != 0 {
		return nil, fmt.Errorf("position list: %s (code %d)", resp.RetMsg, resp.RetCode)
	}
	var result struct {
		List []struct {
			Symbol   string `json:"symbol"`
			Side     string `json:"side"`
			Size     string `json:"size"`
			AvgPrice string `json:"avgPrice"`
		} `json:"list"`
	}
	if err := json.Unmarshal(resp.Result, &result); err != nil {
		return nil, err
	}
	out := make([]PositionInfo, 0, len(result.List))
	for _, item := range result.List {
		size := parseFloatOr(item.Size, 0)
		if size <= 0 {
			continue
		}
		out = append(out, PositionInfo{
			Symbol:   item.Symbol,
			Side:     item.Side,
			Size:     size,
			AvgPrice: parseFloatOr(item.AvgPrice, 0),
		})
	}
	return out, nil
}

func (c *Client) GetOrderRealtime(ctx context.Context, symbol, orderID string) (OrderInfo, error) {
	q := url.Values{
		"category": {"linear"},
		"symbol":   {symbol},
		"orderId":  {orderID},
	}
	var resp APIResponse
	if err := c.signedGet(ctx, "/v5/order/realtime", q, &resp); err != nil {
		return OrderInfo{}, err
	}
	if resp.RetCode != 0 {
		return OrderInfo{}, fmt.Errorf("order realtime: %s (code %d)", resp.RetMsg, resp.RetCode)
	}
	var result struct {
		List []struct {
			OrderID     string `json:"orderId"`
			OrderStatus string `json:"orderStatus"`
			AvgPrice    string `json:"avgPrice"`
			CumExecQty  string `json:"cumExecQty"`
			Price       string `json:"price"`
			Side        string `json:"side"`
			Qty         string `json:"qty"`
			ReduceOnly  bool   `json:"reduceOnly"`
		} `json:"list"`
	}
	if err := json.Unmarshal(resp.Result, &result); err != nil {
		return OrderInfo{}, err
	}
	if len(result.List) == 0 {
		return OrderInfo{OrderID: orderID}, nil
	}
	item := result.List[0]
	return OrderInfo{
		OrderID:     item.OrderID,
		OrderStatus: item.OrderStatus,
		AvgPrice:    parseFloatOr(item.AvgPrice, 0),
		CumExecQty:  parseFloatOr(item.CumExecQty, 0),
		Price:       parseFloatOr(item.Price, 0),
		Side:        item.Side,
		Qty:         parseFloatOr(item.Qty, 0),
		ReduceOnly:  item.ReduceOnly,
	}, nil
}

func (c *Client) CancelOrder(ctx context.Context, symbol, orderID string) error {
	payload := map[string]string{
		"category": "linear",
		"symbol":   symbol,
		"orderId":  orderID,
	}
	body, _ := json.Marshal(payload)
	var resp APIResponse
	if err := c.signedPost(ctx, "/v5/order/cancel", body, &resp); err != nil {
		return err
	}
	if resp.RetCode != 0 {
		return fmt.Errorf("cancel order: %s (code %d)", resp.RetMsg, resp.RetCode)
	}
	return nil
}

func (c *Client) GetRecentClosedPnL(ctx context.Context, symbol string, sinceMs int64) (*ClosedPnLRecord, error) {
	q := url.Values{
		"category":  {"linear"},
		"symbol":    {symbol},
		"startTime": {strconv.FormatInt(sinceMs, 10)},
		"limit":     {"5"},
	}
	var resp APIResponse
	if err := c.signedGet(ctx, "/v5/position/closed-pnl", q, &resp); err != nil {
		return nil, err
	}
	if resp.RetCode != 0 {
		return nil, fmt.Errorf("closed-pnl: %s (code %d)", resp.RetMsg, resp.RetCode)
	}
	var result struct {
		List []struct {
			Symbol        string `json:"symbol"`
			ClosedPnL     string `json:"closedPnl"`
			AvgEntryPrice string `json:"avgEntryPrice"`
			AvgExitPrice  string `json:"avgExitPrice"`
			Qty           string `json:"qty"`
			UpdatedTime   string `json:"updatedTime"`
		} `json:"list"`
	}
	if err := json.Unmarshal(resp.Result, &result); err != nil {
		return nil, err
	}
	if len(result.List) == 0 {
		return nil, nil
	}
	// Return the most recent record.
	best := result.List[0]
	bestTs := parseInt64Or(best.UpdatedTime, 0)
	for _, item := range result.List[1:] {
		ts := parseInt64Or(item.UpdatedTime, 0)
		if ts > bestTs {
			best = item
			bestTs = ts
		}
	}
	return &ClosedPnLRecord{
		Symbol:        best.Symbol,
		ClosedPnL:     parseFloatOr(best.ClosedPnL, 0),
		AvgEntryPrice: parseFloatOr(best.AvgEntryPrice, 0),
		AvgExitPrice:  parseFloatOr(best.AvgExitPrice, 0),
		Qty:           parseFloatOr(best.Qty, 0),
		UpdatedTime:   bestTs,
	}, nil
}

func (c *Client) WaitForClosedPnL(ctx context.Context, symbol string, sinceMs int64, attempts int) (*ClosedPnLRecord, error) {
	for i := 0; i < attempts; i++ {
		rec, err := c.GetRecentClosedPnL(ctx, symbol, sinceMs)
		if err != nil {
			return nil, err
		}
		if rec != nil && rec.UpdatedTime >= sinceMs {
			return rec, nil
		}
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		case <-time.After(300 * time.Millisecond):
		}
	}
	return c.GetRecentClosedPnL(ctx, symbol, sinceMs)
}

func parseInt64Or(s string, def int64) int64 {
	if s == "" {
		return def
	}
	v, err := strconv.ParseInt(s, 10, 64)
	if err != nil {
		return def
	}
	return v
}
