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

// FindNearestResistance finds the nearest ask-side level above refPrice.
// Returns the closest level regardless of size — for scalping, distance matters more than depth.
func FindNearestResistance(ob models.OrderbookSnapshot, refPrice float64) Wall {
	bestDist := math.MaxFloat64
	var best Wall
	for _, l := range ob.Asks {
		p, _ := strconv.ParseFloat(l.Price, 64)
		s, _ := strconv.ParseFloat(l.Size, 64)
		if p <= refPrice || s <= 0 {
			continue
		}
		dist := p - refPrice
		if dist < bestDist {
			bestDist = dist
			best = Wall{Price: p, Size: s, Side: "ask"}
		}
	}
	return best
}

// FindNearestSupport finds the nearest bid-side level below refPrice.
// Returns the closest level regardless of size — for scalping, distance matters more than depth.
func FindNearestSupport(ob models.OrderbookSnapshot, refPrice float64) Wall {
	bestDist := math.MaxFloat64
	var best Wall
	for _, l := range ob.Bids {
		p, _ := strconv.ParseFloat(l.Price, 64)
		s, _ := strconv.ParseFloat(l.Size, 64)
		if p >= refPrice || s <= 0 {
			continue
		}
		dist := refPrice - p
		if dist < bestDist {
			bestDist = dist
			best = Wall{Price: p, Size: s, Side: "bid"}
		}
	}
	return best
}

// FindStrongestResistance finds the ask-side level with largest size above refPrice.
func FindStrongestResistance(ob models.OrderbookSnapshot, refPrice float64) Wall {
	var best Wall
	for _, l := range ob.Asks {
		p, _ := strconv.ParseFloat(l.Price, 64)
		s, _ := strconv.ParseFloat(l.Size, 64)
		if p <= refPrice || s <= 0 {
			continue
		}
		if s > best.Size {
			best = Wall{Price: p, Size: s, Side: "ask"}
		}
	}
	return best
}

// FindStrongestSupport finds the bid-side level with largest size below refPrice.
func FindStrongestSupport(ob models.OrderbookSnapshot, refPrice float64) Wall {
	var best Wall
	for _, l := range ob.Bids {
		p, _ := strconv.ParseFloat(l.Price, 64)
		s, _ := strconv.ParseFloat(l.Size, 64)
		if p >= refPrice || s <= 0 {
			continue
		}
		if s > best.Size {
			best = Wall{Price: p, Size: s, Side: "bid"}
		}
	}
	return best
}

// FindStrongestWallWithin finds the strongest S/R wall within maxDist of refPrice.
// For LONG, looks for resistance above (where price might stall).
// For SHORT, looks for support below (where price might bounce).
func FindStrongestWallWithin(ob models.OrderbookSnapshot, direction string, refPrice, maxDist float64) Wall {
	bestSize := 0.0
	var best Wall

	if direction == "LONG" {
		high := refPrice + maxDist
		for _, l := range ob.Asks {
			p, _ := strconv.ParseFloat(l.Price, 64)
			s, _ := strconv.ParseFloat(l.Size, 64)
			if p <= refPrice || p > high || s <= 0 {
				continue
			}
			if s > bestSize {
				bestSize = s
				best = Wall{Price: p, Size: s, Side: "ask"}
			}
		}
	} else {
		low := refPrice - maxDist
		for _, l := range ob.Bids {
			p, _ := strconv.ParseFloat(l.Price, 64)
			s, _ := strconv.ParseFloat(l.Size, 64)
			if p >= refPrice || p < low || s <= 0 {
				continue
			}
			if s > bestSize {
				bestSize = s
				best = Wall{Price: p, Size: s, Side: "bid"}
			}
		}
	}
	return best
}

// DetectKnife detects rapid adverse price movement conditions.
// Returns (isKnife, tightenFactor). tightenFactor < 1.0 means tighten SL.
func DetectKnife(macroTrend5m, macroTrend15m, obi float64, regime, direction string) (bool, float64) {
	knifeScore := 0.0

	if direction == "LONG" {
		if macroTrend5m < -0.003 {
			knifeScore += 0.4
		}
		if macroTrend15m < -0.008 {
			knifeScore += 0.3
		}
		if obi < -0.2 {
			knifeScore += 0.2
		}
		if regime == "Breakout" && macroTrend5m < -0.002 {
			knifeScore += 0.1
		}
	} else {
		if macroTrend5m > 0.003 {
			knifeScore += 0.4
		}
		if macroTrend15m > 0.008 {
			knifeScore += 0.3
		}
		if obi > 0.2 {
			knifeScore += 0.2
		}
		if regime == "Breakout" && macroTrend5m > 0.002 {
			knifeScore += 0.1
		}
	}

	if knifeScore >= 0.5 {
		tighten := math.Max(0.3, 1.0-knifeScore*0.8)
		return true, tighten
	}
	return false, 1.0
}
