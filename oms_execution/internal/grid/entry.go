package grid

import (
	"math"

	"github.com/fast-trader-gru/oms_execution/internal/liquidity"
	"github.com/fast-trader-gru/oms_execution/internal/models"
)

// AggressiveMakerEntry places PostOnly limits at the spread edge (best bid/ask ± ticks).
// LONG → best bid (+ ticks toward ask, capped below ask); SHORT → best ask (− ticks toward bid).
// Returns 0 when makerTicks <= 0 (caller should use legacy mid-offset entry).
func AggressiveMakerEntry(
	direction string,
	ob models.OrderbookSnapshot,
	tickSize float64,
	makerTicks int,
) float64 {
	if makerTicks <= 0 {
		return 0
	}
	if tickSize <= 0 {
		tickSize = 0.0001
	}
	offset := float64(makerTicks) * tickSize
	bid := liquidity.BestBid(ob)
	ask := liquidity.BestAsk(ob)

	switch direction {
	case "LONG":
		if bid <= 0 {
			return 0
		}
		price := bid + offset
		if ask > 0 && price >= ask {
			price = ask - tickSize
		}
		if price <= 0 {
			price = bid
		}
		return roundToTick(price, tickSize)
	case "SHORT":
		if ask <= 0 {
			return 0
		}
		price := ask - offset
		if bid > 0 && price <= bid {
			price = bid + tickSize
		}
		if price <= 0 {
			price = ask
		}
		return roundToTick(price, tickSize)
	default:
		return 0
	}
}

// PassiveMakerExitPrice returns a reduce-only PostOnly-friendly price at the spread edge.
// LONG close (Sell): best ask − ticks. SHORT close (Buy): best bid + ticks.
func PassiveMakerExitPrice(
	direction string,
	ob models.OrderbookSnapshot,
	tickSize float64,
	makerTicks int,
) float64 {
	switch direction {
	case "LONG":
		return AggressiveMakerEntry("SHORT", ob, tickSize, makerTicks)
	case "SHORT":
		return AggressiveMakerEntry("LONG", ob, tickSize, makerTicks)
	default:
		return 0
	}
}

func CapVolMultiplier(vm, cap float64) float64 {
	if vm <= 0 {
		vm = 1.0
	}
	if cap > 0 && vm > cap {
		vm = cap
	}
	return math.Max(vm, 0.5)
}
