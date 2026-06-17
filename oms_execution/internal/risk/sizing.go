package risk

import "math"

// QtyFromNotional converts USD notional at mark price into contract quantity.
func QtyFromNotional(notionalUSD, markPrice float64) float64 {
	if notionalUSD <= 0 || markPrice <= 0 {
		return 0
	}
	return notionalUSD / markPrice
}

// TradeNotionalUSD is position value: margin × leverage.
func TradeNotionalUSD(marginUSD float64, leverage int) float64 {
	if marginUSD <= 0 || leverage <= 0 {
		return 0
	}
	return marginUSD * float64(leverage)
}

// MaxConcurrentTrades returns how many fixed-margin slots fit in the deposit.
func MaxConcurrentTrades(depositUSD, marginPerTradeUSD float64) int {
	if depositUSD <= 0 || marginPerTradeUSD <= 0 {
		return 0
	}
	return int(math.Floor(depositUSD / marginPerTradeUSD))
}
