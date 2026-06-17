package bybit

import "strings"

// IsOrderNotCancelable reports cancel failures where the order is already filled or gone.
func IsOrderNotCancelable(err error) bool {
	if err == nil {
		return false
	}
	msg := strings.ToLower(err.Error())
	return strings.Contains(msg, "110001") ||
		strings.Contains(msg, "order not exists") ||
		strings.Contains(msg, "too late to cancel")
}
