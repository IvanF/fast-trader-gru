package liquidity

import (
	"math"
	"strconv"

	"github.com/fast-trader-gru/oms_execution/internal/models"
)

type Wall struct {
	Price float64
	Size  float64
	Side  string
	Index int
}

// FindLiquidityWall uses peak detection on orderbook depth (vectorized via scipy equivalent logic).
// Go implementation uses local maxima detection without scipy dependency.
func FindLiquidityWall(ob models.OrderbookSnapshot, direction string, entryPrice, rangePct float64) Wall {
	low := entryPrice * (1 - rangePct)
	high := entryPrice * (1 + rangePct)

	if direction == "LONG" {
		prices, sizes := extractSide(ob.Bids, low, high)
		idx := findPeaks(sizes, 0.3)
		if len(idx) == 0 {
			return Wall{Price: entryPrice * 0.99, Side: "bid"}
		}
		best := idx[0]
		for _, i := range idx[1:] {
			if sizes[i] > sizes[best] {
				best = i
			}
		}
		return Wall{Price: prices[best], Size: sizes[best], Side: "bid", Index: best}
	}

	prices, sizes := extractAsks(ob.Asks, low, high)
	idx := findPeaks(sizes, 0.3)
	if len(idx) == 0 {
		return Wall{Price: entryPrice * 1.01, Side: "ask"}
	}
	best := idx[0]
	for _, i := range idx[1:] {
		if sizes[i] > sizes[best] {
			best = i
		}
	}
	return Wall{Price: prices[best], Size: sizes[best], Side: "ask", Index: best}
}

func extractSide(levels []models.OrderbookLevel, low, high float64) ([]float64, []float64) {
	var prices, sizes []float64
	for _, l := range levels {
		p, _ := strconv.ParseFloat(l.Price, 64)
		s, _ := strconv.ParseFloat(l.Size, 64)
		if p >= low && p <= high {
			prices = append(prices, p)
			sizes = append(sizes, s)
		}
	}
	return prices, sizes
}

func extractAsks(levels []models.OrderbookLevel, low, high float64) ([]float64, []float64) {
	var prices, sizes []float64
	for _, l := range levels {
		p, _ := strconv.ParseFloat(l.Price, 64)
		s, _ := strconv.ParseFloat(l.Size, 64)
		if p >= low && p <= high {
			prices = append(prices, p)
			sizes = append(sizes, s)
		}
	}
	return prices, sizes
}

// findPeaks implements scipy.signal.find_peaks prominence logic in pure Go.
func findPeaks(data []float64, prominence float64) []int {
	if len(data) < 3 {
		return nil
	}
	maxVal := 0.0
	for _, v := range data {
		if v > maxVal {
			maxVal = v
		}
	}
	minProm := prominence * maxVal
	var peaks []int
	for i := 1; i < len(data)-1; i++ {
		if data[i] > data[i-1] && data[i] > data[i+1] {
			leftMin := data[i]
			for j := i - 1; j >= 0; j-- {
				if data[j] < leftMin {
					leftMin = data[j]
				}
			}
			rightMin := data[i]
			for j := i + 1; j < len(data); j++ {
				if data[j] < rightMin {
					rightMin = data[j]
				}
			}
			prom := data[i] - math.Max(leftMin, rightMin)
			if prom >= minProm {
				peaks = append(peaks, i)
			}
		}
	}
	return peaks
}

func AdjustSLBehindWall(wall Wall, direction string, tickSize float64) float64 {
	offset := tickSize * 2
	if direction == "LONG" {
		return wall.Price - offset
	}
	return wall.Price + offset
}
