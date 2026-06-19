package bybit

import (
	"bytes"
	"context"
	"crypto/hmac"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strconv"
	"time"
)

const maxRetries = 5

type Client struct {
	baseURL    string
	apiKey     string
	apiSecret  string
	httpClient *http.Client
	recvWindow string
}

func NewClient(baseURL, apiKey, apiSecret string) *Client {
	return &Client{
		baseURL:    baseURL,
		apiKey:     apiKey,
		apiSecret:  apiSecret,
		recvWindow: "5000",
		httpClient: &http.Client{Timeout: 15 * time.Second},
	}
}

type PlaceOrderRequest struct {
	Category    string `json:"category"`
	Symbol      string `json:"symbol"`
	Side        string `json:"side"`
	OrderType   string `json:"orderType"`
	Qty         string `json:"qty"`
	Price       string `json:"price,omitempty"`
	TimeInForce string `json:"timeInForce"`
	ReduceOnly  bool   `json:"reduceOnly,omitempty"`
	PositionIdx int    `json:"positionIdx"`
}

type APIResponse struct {
	RetCode int             `json:"retCode"`
	RetMsg  string          `json:"retMsg"`
	Result  json.RawMessage `json:"result"`
}

func (c *Client) PlaceLimitOrder(ctx context.Context, req PlaceOrderRequest) (string, error) {
	req.Category = "linear"
	req.OrderType = "Limit"
	req.TimeInForce = "PostOnly"
	body, err := json.Marshal(req)
	if err != nil {
		return "", err
	}
	var resp APIResponse
	if err := c.signedPost(ctx, "/v5/order/create", body, &resp); err != nil {
		return "", err
	}
	if resp.RetCode != 0 {
		return "", fmt.Errorf("place order: %s (code %d)", resp.RetMsg, resp.RetCode)
	}
	var result struct {
		OrderID string `json:"orderId"`
	}
	if err := json.Unmarshal(resp.Result, &result); err != nil {
		return "", err
	}
	return result.OrderID, nil
}

// PlaceReduceLimit posts a GTC reduce-only limit exit (maker-friendly, no market).
func (c *Client) PlaceReduceLimit(ctx context.Context, symbol, side string, qty, qtyStep float64, price string) (string, error) {
	req := PlaceOrderRequest{
		Category:    "linear",
		Symbol:      symbol,
		Side:        side,
		OrderType:   "Limit",
		Qty:         FormatQty(qty, qtyStep),
		Price:       price,
		TimeInForce: "GTC",
		ReduceOnly:  true,
		PositionIdx: 0,
	}
	body, err := json.Marshal(req)
	if err != nil {
		return "", err
	}
	var resp APIResponse
	if err := c.signedPost(ctx, "/v5/order/create", body, &resp); err != nil {
		return "", err
	}
	if resp.RetCode != 0 {
		return "", fmt.Errorf("reduce limit: %s (code %d)", resp.RetMsg, resp.RetCode)
	}
	var result struct {
		OrderID string `json:"orderId"`
	}
	if err := json.Unmarshal(resp.Result, &result); err != nil {
		return "", err
	}
	return result.OrderID, nil
}

// PlaceStopMarket posts a conditional stop-market reduce-only order.
// triggerDirection: 1 = price rises to trigger, 2 = price falls to trigger.
func (c *Client) PlaceStopMarket(ctx context.Context, symbol, side string, qty, qtyStep float64, triggerPrice string, triggerDirection int) (string, error) {
	req := map[string]interface{}{
		"category":         "linear",
		"symbol":           symbol,
		"side":             side,
		"orderType":        "Market",
		"qty":              FormatQty(qty, qtyStep),
		"triggerPrice":     triggerPrice,
		"triggerDirection": triggerDirection,
		"reduceOnly":       true,
		"positionIdx":      0,
	}
	body, err := json.Marshal(req)
	if err != nil {
		return "", err
	}
	var resp APIResponse
	if err := c.signedPost(ctx, "/v5/order/create", body, &resp); err != nil {
		return "", err
	}
	if resp.RetCode != 0 {
		return "", fmt.Errorf("stop market: %s (code %d)", resp.RetMsg, resp.RetCode)
	}
	var result struct {
		OrderID string `json:"orderId"`
	}
	if err := json.Unmarshal(resp.Result, &result); err != nil {
		return "", err
	}
	return result.OrderID, nil
}

// PlaceReducePostOnlyLimit posts a reduce-only PostOnly limit (never crosses the spread as taker).
func (c *Client) PlaceReducePostOnlyLimit(ctx context.Context, symbol, side string, qty, qtyStep float64, price string) (string, error) {
	req := PlaceOrderRequest{
		Category:    "linear",
		Symbol:      symbol,
		Side:        side,
		OrderType:   "Limit",
		Qty:         FormatQty(qty, qtyStep),
		Price:       price,
		TimeInForce: "PostOnly",
		ReduceOnly:  true,
		PositionIdx: 0,
	}
	body, err := json.Marshal(req)
	if err != nil {
		return "", err
	}
	var resp APIResponse
	if err := c.signedPost(ctx, "/v5/order/create", body, &resp); err != nil {
		return "", err
	}
	if resp.RetCode != 0 {
		return "", fmt.Errorf("reduce postonly limit: %s (code %d)", resp.RetMsg, resp.RetCode)
	}
	var result struct {
		OrderID string `json:"orderId"`
	}
	if err := json.Unmarshal(resp.Result, &result); err != nil {
		return "", err
	}
	return result.OrderID, nil
}

func (c *Client) PlaceMarketOrder(ctx context.Context, symbol, side string, qty, qtyStep float64, reduceOnly bool) (string, error) {
	req := PlaceOrderRequest{
		Category:    "linear",
		Symbol:      symbol,
		Side:        side,
		OrderType:   "Market",
		Qty:         FormatQty(qty, qtyStep),
		ReduceOnly:  reduceOnly,
		PositionIdx: 0,
	}
	body, err := json.Marshal(req)
	if err != nil {
		return "", err
	}
	var resp APIResponse
	if err := c.signedPost(ctx, "/v5/order/create", body, &resp); err != nil {
		return "", err
	}
	if resp.RetCode != 0 {
		return "", fmt.Errorf("market order: %s (code %d)", resp.RetMsg, resp.RetCode)
	}
	var result struct {
		OrderID string `json:"orderId"`
	}
	_ = json.Unmarshal(resp.Result, &result)
	return result.OrderID, nil
}

func (c *Client) CancelAllOrders(ctx context.Context, symbol string) error {
	payload := map[string]string{
		"category": "linear",
		"symbol":   symbol,
	}
	body, _ := json.Marshal(payload)
	var resp APIResponse
	if err := c.signedPost(ctx, "/v5/order/cancel-all", body, &resp); err != nil {
		return err
	}
	if resp.RetCode != 0 {
		return fmt.Errorf("cancel-all: %s (code %d)", resp.RetMsg, resp.RetCode)
	}
	return nil
}

// CancelAllConditionalOrders cancels all conditional (stop-market, etc.) orders for a symbol.
func (c *Client) CancelAllConditionalOrders(ctx context.Context, symbol string) error {
	payload := map[string]string{
		"category": "conditional",
		"symbol":   symbol,
	}
	body, _ := json.Marshal(payload)
	var resp APIResponse
	if err := c.signedPost(ctx, "/v5/order/cancel-all", body, &resp); err != nil {
		return err
	}
	if resp.RetCode != 0 {
		return fmt.Errorf("cancel-all conditional: %s (code %d)", resp.RetMsg, resp.RetCode)
	}
	return nil
}

func (c *Client) SetLeverage(ctx context.Context, symbol string, leverage int) error {
	if leverage <= 0 {
		return fmt.Errorf("leverage must be > 0")
	}
	lev := strconv.Itoa(leverage)
	payload := map[string]string{
		"category":    "linear",
		"symbol":      symbol,
		"buyLeverage": lev,
		"sellLeverage": lev,
	}
	body, err := json.Marshal(payload)
	if err != nil {
		return err
	}
	var resp APIResponse
	if err := c.signedPost(ctx, "/v5/position/set-leverage", body, &resp); err != nil {
		return err
	}
	// 110043: leverage not modified (already set)
	if resp.RetCode != 0 && resp.RetCode != 110043 {
		return fmt.Errorf("set leverage: %s (code %d)", resp.RetMsg, resp.RetCode)
	}
	return nil
}

func (c *Client) GetInstrument(ctx context.Context, symbol string) (InstrumentInfo, error) {
	url := fmt.Sprintf("%s/v5/market/instruments-info?category=linear&symbol=%s", c.baseURL, symbol)
	var resp struct {
		RetCode int    `json:"retCode"`
		RetMsg  string `json:"retMsg"`
		Result  struct {
			List []struct {
				PriceFilter struct {
					TickSize string `json:"tickSize"`
				} `json:"priceFilter"`
				LotSizeFilter struct {
					QtyStep     string `json:"qtyStep"`
					MinOrderQty string `json:"minOrderQty"`
					MaxOrderQty string `json:"maxOrderQty"`
				} `json:"lotSizeFilter"`
			} `json:"list"`
		} `json:"result"`
	}
	if err := c.getWithRetry(ctx, url, &resp); err != nil {
		return InstrumentInfo{}, err
	}
	if resp.RetCode != 0 {
		return InstrumentInfo{}, fmt.Errorf("instruments-info: %s (code %d)", resp.RetMsg, resp.RetCode)
	}
	if len(resp.Result.List) == 0 {
		return InstrumentInfo{
			TickSize: 0.01,
			Lot:      LotFilters{QtyStep: 0.001, MinOrderQty: 0.001, MaxOrderQty: 1e9},
		}, nil
	}
	item := resp.Result.List[0]
	info := InstrumentInfo{
		TickSize: parseFloatOr(item.PriceFilter.TickSize, 0.01),
		Lot: LotFilters{
			QtyStep:     parseFloatOr(item.LotSizeFilter.QtyStep, 0.001),
			MinOrderQty: parseFloatOr(item.LotSizeFilter.MinOrderQty, 0.001),
			MaxOrderQty: parseFloatOr(item.LotSizeFilter.MaxOrderQty, 1e9),
		},
	}
	if info.Lot.QtyStep <= 0 {
		info.Lot.QtyStep = 0.001
	}
	if info.Lot.MinOrderQty <= 0 {
		info.Lot.MinOrderQty = info.Lot.QtyStep
	}
	return info, nil
}

// GetInstrumentInfo is kept for backward compatibility.
func (c *Client) GetInstrumentInfo(ctx context.Context, symbol string) (tickSize, lotSize float64, err error) {
	info, err := c.GetInstrument(ctx, symbol)
	if err != nil {
		return 0, 0, err
	}
	return info.TickSize, info.Lot.QtyStep, nil
}

func parseFloatOr(s string, def float64) float64 {
	if s == "" {
		return def
	}
	v, err := strconv.ParseFloat(s, 64)
	if err != nil || v == 0 {
		return def
	}
	return v
}

func (c *Client) signedPost(ctx context.Context, path string, body []byte, dest *APIResponse) error {
	timestamp := strconv.FormatInt(time.Now().UnixMilli(), 10)
	signPayload := timestamp + c.apiKey + c.recvWindow + string(body)
	mac := hmac.New(sha256.New, []byte(c.apiSecret))
	mac.Write([]byte(signPayload))
	signature := hex.EncodeToString(mac.Sum(nil))

	url := c.baseURL + path
	backoff := 500 * time.Millisecond
	var lastErr error
	for i := 0; i < maxRetries; i++ {
		req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(body))
		if err != nil {
			return err
		}
		req.Header.Set("Content-Type", "application/json")
		req.Header.Set("X-BAPI-API-KEY", c.apiKey)
		req.Header.Set("X-BAPI-SIGN", signature)
		req.Header.Set("X-BAPI-TIMESTAMP", timestamp)
		req.Header.Set("X-BAPI-RECV-WINDOW", c.recvWindow)

		res, err := c.httpClient.Do(req)
		if err != nil {
			lastErr = err
			time.Sleep(backoff)
			backoff *= 2
			continue
		}
		respBody, _ := io.ReadAll(res.Body)
		res.Body.Close()

		if res.StatusCode == http.StatusTooManyRequests {
			lastErr = fmt.Errorf("rate limited")
			time.Sleep(backoff)
			backoff *= 2
			continue
		}
		if res.StatusCode < 200 || res.StatusCode >= 300 {
			lastErr = fmt.Errorf("http %d: %s", res.StatusCode, string(respBody))
			time.Sleep(backoff)
			backoff *= 2
			continue
		}
		if err := json.Unmarshal(respBody, dest); err != nil {
			return err
		}
		return nil
	}
	return fmt.Errorf("signed post failed: %w", lastErr)
}

func (c *Client) signedGet(ctx context.Context, path string, query url.Values, dest *APIResponse) error {
	timestamp := strconv.FormatInt(time.Now().UnixMilli(), 10)
	qs := query.Encode()
	signPayload := timestamp + c.apiKey + c.recvWindow + qs
	mac := hmac.New(sha256.New, []byte(c.apiSecret))
	mac.Write([]byte(signPayload))
	signature := hex.EncodeToString(mac.Sum(nil))

	reqURL := c.baseURL + path
	if qs != "" {
		reqURL += "?" + qs
	}

	backoff := 500 * time.Millisecond
	var lastErr error
	for i := 0; i < maxRetries; i++ {
		req, err := http.NewRequestWithContext(ctx, http.MethodGet, reqURL, nil)
		if err != nil {
			return err
		}
		req.Header.Set("X-BAPI-API-KEY", c.apiKey)
		req.Header.Set("X-BAPI-SIGN", signature)
		req.Header.Set("X-BAPI-TIMESTAMP", timestamp)
		req.Header.Set("X-BAPI-RECV-WINDOW", c.recvWindow)

		res, err := c.httpClient.Do(req)
		if err != nil {
			lastErr = err
			time.Sleep(backoff)
			backoff *= 2
			continue
		}
		respBody, _ := io.ReadAll(res.Body)
		res.Body.Close()

		if res.StatusCode == http.StatusTooManyRequests {
			lastErr = fmt.Errorf("rate limited")
			time.Sleep(backoff)
			backoff *= 2
			continue
		}
		if res.StatusCode < 200 || res.StatusCode >= 300 {
			lastErr = fmt.Errorf("http %d: %s", res.StatusCode, string(respBody))
			time.Sleep(backoff)
			backoff *= 2
			continue
		}
		if err := json.Unmarshal(respBody, dest); err != nil {
			return err
		}
		return nil
	}
	return fmt.Errorf("signed get failed: %w", lastErr)
}

func (c *Client) getWithRetry(ctx context.Context, url string, dest any) error {
	backoff := 500 * time.Millisecond
	var lastErr error
	for i := 0; i < maxRetries; i++ {
		req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
		if err != nil {
			return err
		}
		res, err := c.httpClient.Do(req)
		if err != nil {
			lastErr = err
			time.Sleep(backoff)
			backoff *= 2
			continue
		}
		body, _ := io.ReadAll(res.Body)
		res.Body.Close()
		if res.StatusCode == http.StatusTooManyRequests {
			lastErr = fmt.Errorf("rate limited")
			time.Sleep(backoff)
			backoff *= 2
			continue
		}
		if err := json.Unmarshal(body, dest); err != nil {
			return err
		}
		return nil
	}
	return lastErr
}

func formatQty(q float64) string {
	return FormatQty(q, 0)
}

func FormatPrice(p float64) string {
	return strconv.FormatFloat(p, 'f', -1, 64)
}
