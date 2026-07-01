package liquidity

import (
	"math"
	"sort"
	"strconv"

	"github.com/fast-trader-gru/oms_execution/internal/models"
)

type obLevel struct {
	price float64
	size  float64
}

// LiquiditySLResult contains the computed SL price and metadata.
type LiquiditySLResult struct {
	Price          float64
	Distance       float64
	DistancePct    float64
	LevelPrice     float64 // the S/R level behind which SL is placed
	LevelSize      float64 // volume at that level
	Source         string  // "nearest_level", "cluster", "min_sl_fallback"
}

// ComputeLiquiditySL scans the orderbook in the adverse direction and places SL
// just beyond the nearest liquidity zone where a bounce is expected.
//
// Logic:
// 1. Scan levels from entry outward, group nearby levels (within 0.5% of each other) into density zones
// 2. Find the NEAREST zone with cumulative size > threshold — that's the bounce point
// 3. SL = behind that zone (2 ticks buffer)
// 4. If no zone within maxSL → SL at min distance
func ComputeLiquiditySL(
	direction string,
	fillPrice float64,
	ob models.OrderbookSnapshot,
	tickSize float64,
	minSLPct, maxSLPct float64,
) LiquiditySLResult {
	if fillPrice <= 0 || tickSize <= 0 {
		return LiquiditySLResult{Price: fillPrice, Source: "invalid_input"}
	}

	maxSLDist := fillPrice * maxSLPct
	minSLDist := fillPrice * minSLPct

	if direction == "LONG" {
		return computeLongSL(ob, fillPrice, tickSize, minSLDist, maxSLDist)
	}
	return computeShortSL(ob, fillPrice, tickSize, minSLDist, maxSLDist)
}

// computeLongSL finds support below entry and places SL just below it.
func computeLongSL(ob models.OrderbookSnapshot, fillPrice, tickSize, minSLDist, maxSLDist float64) LiquiditySLResult {
	var levels []obLevel
	for _, l := range ob.Bids {
		p, _ := strconv.ParseFloat(l.Price, 64)
		s, _ := strconv.ParseFloat(l.Size, 64)
		if p < fillPrice && p > 0 && s > 0 {
			levels = append(levels, obLevel{price: p, size: s})
		}
	}
	sort.Slice(levels, func(i, j int) bool { return levels[i].price > levels[j].price })

	if len(levels) == 0 {
		sl := roundToTick(fillPrice-minSLDist, tickSize)
		return LiquiditySLResult{Price: sl, Distance: minSLDist, DistancePct: minSLDist / fillPrice, Source: "min_sl_fallback"}
	}

	// Build density zones: group levels within 0.5% of each other
	type zone struct {
		topPrice    float64 // price nearest to entry
		bottomPrice float64 // price furthest from entry
		totalSize   float64
		count       int
	}
	var zones []zone
	var current *zone
	for _, lv := range levels {
		dist := fillPrice - lv.price
		if dist > maxSLDist {
			break
		}
		if current == nil {
			z := zone{topPrice: lv.price, bottomPrice: lv.price, totalSize: lv.size, count: 1}
			current = &z
			continue
		}
		// Check if this level is within 1.5% of the zone top (wider grouping for thin books)
		zoneRangePct := (current.topPrice - lv.price) / fillPrice
		if zoneRangePct <= 0.015 {
			current.bottomPrice = lv.price
			current.totalSize += lv.size
			current.count++
		} else {
			zones = append(zones, *current)
			z := zone{topPrice: lv.price, bottomPrice: lv.price, totalSize: lv.size, count: 1}
			current = &z
		}
	}
	if current != nil {
		zones = append(zones, *current)
	}

	// Find the nearest zone where SL behind it would be at least minSLDist away.
	// This places SL at a real support level while respecting the minimum distance.
	if len(zones) > 0 {
		var bestZone *zone
		for i := range zones {
			z := &zones[i]
			slCandidate := z.bottomPrice - tickSize*2
			slDist := fillPrice - slCandidate
			if slDist >= minSLDist {
				bestZone = z
				break
			}
		}
		// If no zone qualifies, use the nearest zone anyway (real S/R > artificial floor)
		if bestZone == nil {
			bestZone = &zones[0]
		}
		sl := bestZone.bottomPrice - tickSize*2

		slDist := fillPrice - sl
		if slDist < minSLDist {
			sl = fillPrice - minSLDist
		}
		if sl > fillPrice {
			sl = fillPrice - minSLDist
		}

		sl = roundToTick(sl, tickSize)
		return LiquiditySLResult{
			Price:       sl,
			Distance:    fillPrice - sl,
			DistancePct: (fillPrice - sl) / fillPrice,
			LevelPrice:  bestZone.bottomPrice,
			LevelSize:   bestZone.totalSize,
			Source:      "liquidity_zone",
		}
	}

	// Fallback: use nearest level directly
	nearest := levels[0]
	sl := nearest.price - tickSize*2
	slDist := fillPrice - sl
	if slDist < minSLDist {
		sl = fillPrice - minSLDist
	}
	if sl > fillPrice {
		sl = fillPrice - minSLDist
	}
	sl = roundToTick(sl, tickSize)
	return LiquiditySLResult{
		Price:       sl,
		Distance:    fillPrice - sl,
		DistancePct: (fillPrice - sl) / fillPrice,
		LevelPrice:  nearest.price,
		LevelSize:   nearest.size,
		Source:      "nearest_level",
	}
}

// computeShortSL finds resistance above entry and places SL just above it.
func computeShortSL(ob models.OrderbookSnapshot, fillPrice, tickSize, minSLDist, maxSLDist float64) LiquiditySLResult {
	var levels []obLevel
	for _, l := range ob.Asks {
		p, _ := strconv.ParseFloat(l.Price, 64)
		s, _ := strconv.ParseFloat(l.Size, 64)
		if p > fillPrice && s > 0 {
			levels = append(levels, obLevel{price: p, size: s})
		}
	}
	sort.Slice(levels, func(i, j int) bool { return levels[i].price < levels[j].price })

	if len(levels) == 0 {
		sl := roundToTick(fillPrice+minSLDist, tickSize)
		return LiquiditySLResult{Price: sl, Distance: minSLDist, DistancePct: minSLDist / fillPrice, Source: "min_sl_fallback"}
	}

	type zone struct {
		topPrice    float64
		bottomPrice float64
		totalSize   float64
		count       int
	}
	var zones []zone
	var current *zone
	for _, lv := range levels {
		dist := lv.price - fillPrice
		if dist > maxSLDist {
			break
		}
		if current == nil {
			z := zone{topPrice: lv.price, bottomPrice: lv.price, totalSize: lv.size, count: 1}
			current = &z
			continue
		}
		zoneRangePct := (lv.price - current.bottomPrice) / fillPrice
		if zoneRangePct <= 0.015 {
			current.topPrice = lv.price
			current.totalSize += lv.size
			current.count++
		} else {
			zones = append(zones, *current)
			z := zone{topPrice: lv.price, bottomPrice: lv.price, totalSize: lv.size, count: 1}
			current = &z
		}
	}
	if current != nil {
		zones = append(zones, *current)
	}

	// Find the nearest zone where SL behind it would be at least minSLDist away.
	if len(zones) > 0 {
		var bestZone *zone
		for i := range zones {
			z := &zones[i]
			slCandidate := z.topPrice + tickSize*2
			slDist := slCandidate - fillPrice
			if slDist >= minSLDist {
				bestZone = z
				break
			}
		}
		if bestZone == nil {
			bestZone = &zones[0]
		}
		sl := bestZone.topPrice + tickSize*2

		slDist := sl - fillPrice
		if slDist < minSLDist {
			sl = fillPrice + minSLDist
		}
		if sl < fillPrice {
			sl = fillPrice + minSLDist
		}

		sl = roundToTick(sl, tickSize)
		return LiquiditySLResult{
			Price:       sl,
			Distance:    sl - fillPrice,
			DistancePct: (sl - fillPrice) / fillPrice,
			LevelPrice:  bestZone.topPrice,
			LevelSize:   bestZone.totalSize,
			Source:      "liquidity_zone",
		}
	}

	nearest := levels[0]
	sl := nearest.price + tickSize*2
	slDist := sl - fillPrice
	if slDist < minSLDist {
		sl = fillPrice + minSLDist
	}
	if sl < fillPrice {
		sl = fillPrice + minSLDist
	}
	sl = roundToTick(sl, tickSize)
	return LiquiditySLResult{
		Price:       sl,
		Distance:    sl - fillPrice,
		DistancePct: (sl - fillPrice) / fillPrice,
		LevelPrice:  nearest.price,
		LevelSize:   nearest.size,
		Source:      "nearest_level",
	}
}

func roundToTick(price, tickSize float64) float64 {
	if tickSize <= 0 {
		return price
	}
	return math.Round(price/tickSize) * tickSize
}
