package bybit

import (
	"context"
	"encoding/json"
	"net/url"
)

// OpenOrder is a live linear order on the exchange.
type OpenOrder struct {
	OrderID     string
	Symbol      string
	Side        string
	OrderStatus string
	Price       float64
	Qty         float64
	ReduceOnly  bool
}

// ListOpenOrders returns all open linear USDT-margined orders.
func (c *Client) ListOpenOrders(ctx context.Context) ([]OpenOrder, error) {
	q := url.Values{
		"category":   {"linear"},
		"settleCoin": {"USDT"},
		"openOnly":   {"0"},
		"limit":      {"50"},
	}
	var resp APIResponse
	if err := c.signedGet(ctx, "/v5/order/realtime", q, &resp); err != nil {
		return nil, err
	}
	if resp.RetCode != 0 {
		return nil, errFromAPI("order realtime", resp)
	}
	var result struct {
		List []struct {
			OrderID     string `json:"orderId"`
			Symbol      string `json:"symbol"`
			Side        string `json:"side"`
			OrderStatus string `json:"orderStatus"`
			Price       string `json:"price"`
			Qty         string `json:"qty"`
			ReduceOnly  bool   `json:"reduceOnly"`
		} `json:"list"`
	}
	if err := json.Unmarshal(resp.Result, &result); err != nil {
		return nil, err
	}
	out := make([]OpenOrder, 0, len(result.List))
	for _, item := range result.List {
		if item.OrderStatus == "Filled" || item.OrderStatus == "Cancelled" {
			continue
		}
		out = append(out, OpenOrder{
			OrderID:     item.OrderID,
			Symbol:      item.Symbol,
			Side:        item.Side,
			OrderStatus: item.OrderStatus,
			Price:       parseFloatOr(item.Price, 0),
			Qty:         parseFloatOr(item.Qty, 0),
			ReduceOnly:  item.ReduceOnly,
		})
	}
	return out, nil
}

func errFromAPI(op string, resp APIResponse) error {
	return &APIError{Op: op, Code: resp.RetCode, Msg: resp.RetMsg}
}

type APIError struct {
	Op   string
	Code int
	Msg  string
}

func (e *APIError) Error() string {
	return e.Op + ": " + e.Msg
}
