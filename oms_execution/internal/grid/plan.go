package grid

import (
	"math"
	"strconv"

	"github.com/fast-trader-gru/oms_execution/internal/liquidity"
	"github.com/fast-trader-gru/oms_execution/internal/models"
)

const baseGridSpacing = 0.0025

type PlanOptions struct {
	EntryMakerTicks  int
	VolMultiplierCap float64
}

func BuildPlan(signal models.TradeSignal, ob models.OrderbookSnapshot, tickSize float64, qty float64, timeStopSec int, opts PlanOptions) models.GridPlan {
	mid := MidPrice(ob)
	if mid == 0 {
		mid = 1.0
	}

	vm := CapVolMultiplier(signal.VolatilityMultiplier, opts.VolMultiplierCap)
	spacing := baseGridSpacing * vm

	if qty <= 0 {
		qty = 0.001
	}

	direction := signal.Direction
	rangePct := spacing * 4

	wall := liquidity.FindLiquidityWall(ob, direction, mid, rangePct)
	sl := liquidity.AdjustSLBehindWall(wall, direction, tickSize)

	var entry float64
	var tps []float64
	risk := spacing * mid

	if makerEntry := AggressiveMakerEntry(direction, ob, tickSize, opts.EntryMakerTicks); makerEntry > 0 {
		entry = makerEntry
	} else if direction == "LONG" {
		entry = mid - spacing*mid
	} else {
		entry = mid + spacing*mid
	}

	if direction == "LONG" {
		if sl >= entry {
			sl = entry - risk
		}
		tps = []float64{
			entry + risk,
			entry + risk*2,
			entry + risk*3,
		}
	} else {
		if sl <= entry {
			sl = entry + risk
		}
		tps = []float64{
			entry - risk,
			entry - risk*2,
			entry - risk*3,
		}
	}

	entry = roundToTick(entry, tickSize)
	sl = roundToTick(sl, tickSize)
	for i := range tps {
		tps[i] = roundToTick(tps[i], tickSize)
	}

	return models.GridPlan{
		Symbol:      signal.Symbol,
		Direction:   direction,
		EntryPrice:  entry,
		StopLoss:    sl,
		TakeProfits: tps,
		Qty:         qty,
		TimeStopSec: timeStopSec,
		Signal:      signal,
		WallPrice:   wall.Price,
	}
}

func parseF(s string) (float64, error) {
	return strconv.ParseFloat(s, 64)
}

func roundToTick(price, tick float64) float64 {
	return RoundToTick(price, tick)
}

// RoundToTick rounds price to the nearest exchange tick.
func RoundToTick(price, tick float64) float64 {
	if tick <= 0 {
		return price
	}
	return math.Round(price/tick) * tick
}
