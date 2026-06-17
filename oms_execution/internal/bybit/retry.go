package bybit

import (
	"context"
	"strings"
	"time"
)

const maxMarketCloseRetries = 6

// IsRateLimitError reports whether err is a Bybit HTTP 429 / rate-limit response.
func IsRateLimitError(err error) bool {
	if err == nil {
		return false
	}
	msg := strings.ToLower(err.Error())
	return strings.Contains(msg, "rate limit") || strings.Contains(msg, "429") || strings.Contains(msg, "too many")
}

// PlaceReduceMarketRetry submits a reduce-only market close with exponential backoff on rate limits.
func (c *Client) PlaceReduceMarketRetry(ctx context.Context, symbol, side string, qty, qtyStep float64) (string, error) {
	backoff := 400 * time.Millisecond
	var lastErr error
	for attempt := 0; attempt < maxMarketCloseRetries; attempt++ {
		if ctx.Err() != nil {
			return "", ctx.Err()
		}
		id, err := c.PlaceMarketOrder(ctx, symbol, side, qty, qtyStep, true)
		if err == nil {
			return id, nil
		}
		lastErr = err
		if !IsRateLimitError(err) {
			return "", err
		}
		select {
		case <-ctx.Done():
			return "", ctx.Err()
		case <-time.After(backoff):
		}
		backoff *= 2
		if backoff > 5*time.Second {
			backoff = 5 * time.Second
		}
	}
	return "", lastErr
}
