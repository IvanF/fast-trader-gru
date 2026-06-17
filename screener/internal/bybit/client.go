package bybit

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strconv"
	"time"
)

const (
	maxRetries     = 5
	initialBackoff = 500 * time.Millisecond
)

type Client struct {
	baseURL    string
	httpClient *http.Client
}

func NewClient(baseURL string) *Client {
	return &Client{
		baseURL: baseURL,
		httpClient: &http.Client{
			Timeout: 30 * time.Second,
		},
	}
}

type TickerItem struct {
	Symbol       string `json:"symbol"`
	Turnover24h  string `json:"turnover24h"`
	FundingRate  string `json:"fundingRate"`
	LastPrice    string `json:"lastPrice"`
}

type tickersResponse struct {
	RetCode int    `json:"retCode"`
	RetMsg  string `json:"retMsg"`
	Result  struct {
		List []TickerItem `json:"list"`
	} `json:"result"`
}

type fundingResponse struct {
	RetCode int    `json:"retCode"`
	RetMsg  string `json:"retMsg"`
	Result  struct {
		List []struct {
			Symbol          string `json:"symbol"`
			FundingRate     string `json:"fundingRate"`
			NextFundingTime string `json:"nextFundingTime"`
		} `json:"list"`
	} `json:"result"`
}

func (c *Client) GetLinearTickers(ctx context.Context) ([]TickerItem, error) {
	url := fmt.Sprintf("%s/v5/market/tickers?category=linear", c.baseURL)
	var resp tickersResponse
	if err := c.getWithRetry(ctx, url, &resp); err != nil {
		return nil, err
	}
	if resp.RetCode != 0 {
		return nil, fmt.Errorf("bybit tickers error: %s (code %d)", resp.RetMsg, resp.RetCode)
	}
	return resp.Result.List, nil
}

func (c *Client) GetFundingRates(ctx context.Context) (map[string]float64, error) {
	url := fmt.Sprintf("%s/v5/market/tickers?category=linear", c.baseURL)
	var resp tickersResponse
	if err := c.getWithRetry(ctx, url, &resp); err != nil {
		return nil, err
	}
	if resp.RetCode != 0 {
		return nil, fmt.Errorf("bybit funding error: %s (code %d)", resp.RetMsg, resp.RetCode)
	}
	rates := make(map[string]float64, len(resp.Result.List))
	for _, item := range resp.Result.List {
		if item.FundingRate == "" {
			continue
		}
		r, err := strconv.ParseFloat(item.FundingRate, 64)
		if err != nil {
			continue
		}
		rates[item.Symbol] = r
	}
	return rates, nil
}

func (c *Client) getWithRetry(ctx context.Context, url string, dest any) error {
	backoff := initialBackoff
	var lastErr error
	for attempt := 0; attempt < maxRetries; attempt++ {
		if ctx.Err() != nil {
			return ctx.Err()
		}
		req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
		if err != nil {
			return err
		}
		req.Header.Set("Accept", "application/json")

		res, err := c.httpClient.Do(req)
		if err != nil {
			lastErr = err
			time.Sleep(backoff)
			backoff *= 2
			continue
		}

		body, readErr := io.ReadAll(res.Body)
		res.Body.Close()
		if readErr != nil {
			lastErr = readErr
			time.Sleep(backoff)
			backoff *= 2
			continue
		}

		if res.StatusCode == http.StatusTooManyRequests {
			retryAfter := res.Header.Get("Retry-After")
			wait := backoff
			if retryAfter != "" {
				if secs, err := strconv.Atoi(retryAfter); err == nil {
					wait = time.Duration(secs) * time.Second
				}
			}
			lastErr = fmt.Errorf("rate limited (429)")
			time.Sleep(wait)
			backoff *= 2
			continue
		}

		if res.StatusCode < 200 || res.StatusCode >= 300 {
			lastErr = fmt.Errorf("http %d: %s", res.StatusCode, string(body))
			time.Sleep(backoff)
			backoff *= 2
			continue
		}

		if err := json.Unmarshal(body, dest); err != nil {
			return fmt.Errorf("decode response: %w", err)
		}
		return nil
	}
	return fmt.Errorf("bybit request failed after %d retries: %w", maxRetries, lastErr)
}
