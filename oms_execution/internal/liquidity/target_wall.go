package liquidity

import (
	"strconv"

	"github.com/fast-trader-gru/oms_execution/internal/models"
)

// FindResistanceWall finds the strongest ask-side liquidity peak above refPrice (LONG TP zone).
func FindResistanceWall(ob models.OrderbookSnapshot, refPrice, rangePct float64) Wall {
	low := refPrice
	high := refPrice * (1 + rangePct)
	prices, sizes := extractAsks(ob.Asks, low, high)
	if len(prices) == 0 {
		return Wall{Price: refPrice * (1 + rangePct*0.5), Side: "ask"}
	}
	idx := findPeaks(sizes, 0.3)
	if len(idx) == 0 {
		best := 0
		for i := 1; i < len(sizes); i++ {
			if sizes[i] > sizes[best] {
				best = i
			}
		}
		return Wall{Price: prices[best], Size: sizes[best], Side: "ask", Index: best}
	}
	best := idx[0]
	for _, i := range idx[1:] {
		if sizes[i] > sizes[best] {
			best = i
		}
	}
	return Wall{Price: prices[best], Size: sizes[best], Side: "ask", Index: best}
}

// FindSupportWall finds the strongest bid-side liquidity peak below refPrice (SHORT TP zone).
func FindSupportWall(ob models.OrderbookSnapshot, refPrice, rangePct float64) Wall {
	low := refPrice * (1 - rangePct)
	high := refPrice
	prices, sizes := extractSide(ob.Bids, low, high)
	if len(prices) == 0 {
		return Wall{Price: refPrice * (1 - rangePct*0.5), Side: "bid"}
	}
	idx := findPeaks(sizes, 0.3)
	if len(idx) == 0 {
		best := 0
		for i := 1; i < len(sizes); i++ {
			if sizes[i] > sizes[best] {
				best = i
			}
		}
		return Wall{Price: prices[best], Size: sizes[best], Side: "bid", Index: best}
	}
	best := idx[0]
	for _, i := range idx[1:] {
		if sizes[i] > sizes[best] {
			best = i
		}
	}
	return Wall{Price: prices[best], Size: sizes[best], Side: "bid", Index: best}
}

// BestBid returns the top bid price or 0.
func BestBid(ob models.OrderbookSnapshot) float64 {
	if len(ob.Bids) == 0 {
		return 0
	}
	p, _ := strconv.ParseFloat(ob.Bids[0].Price, 64)
	return p
}

// BestAsk returns the top ask price or 0.
func BestAsk(ob models.OrderbookSnapshot) float64 {
	if len(ob.Asks) == 0 {
		return 0
	}
	p, _ := strconv.ParseFloat(ob.Asks[0].Price, 64)
	return p
}
