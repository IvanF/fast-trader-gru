package executor

import (
	"log/slog"
	"math"
	"strconv"
	"time"

	"github.com/fast-trader-gru/oms_execution/internal/liquidity"
	"github.com/fast-trader-gru/oms_execution/internal/models"
)

const (
	densityWallThreshold  = 15.0
	densityVelocityThresh = 0.4
	densityStallSec       = 90
	densityStallR         = 0.15
	densityTPPushPct      = 0.002
)

type DensityAction struct {
	Type    string
	Symbol  string
	TPPrice float64
	Reason  string
}

type DensityExitManager struct {
	logger *slog.Logger
	obi    *liquidity.OrderbookMomentum
}

func NewDensityExitManager(logger *slog.Logger, _ float64) *DensityExitManager {
	return &DensityExitManager{
		logger: logger,
		obi:    liquidity.NewOrderbookMomentum(20),
	}
}

func (d *DensityExitManager) UpdateOrderbook(symbol string, ob models.OrderbookSnapshot) {
	d.obi.Update(symbol, ob)
}

func (d *DensityExitManager) EvaluateExits(
	pos *models.ActivePosition,
	ob models.OrderbookSnapshot,
	currentPrice float64,
	tickSize float64,
) []DensityAction {
	if pos == nil {
		return nil
	}
	if len(ob.Bids) == 0 || len(ob.Asks) == 0 {
		return nil
	}

	var actions []DensityAction

	if a := d.checkWallPush(pos, ob, currentPrice, tickSize); a != nil {
		actions = append(actions, *a)
	}
	if a := d.checkVelocityReversal(pos); a != nil {
		actions = append(actions, *a)
	}
	if a := d.checkStagnation(pos, currentPrice); a != nil {
		actions = append(actions, *a)
	}

	return actions
}

func (d *DensityExitManager) checkWallPush(
	pos *models.ActivePosition,
	ob models.OrderbookSnapshot,
	currentPrice float64,
	tickSize float64,
) *DensityAction {
	if len(pos.TakeProfitOrders) == 0 {
		return nil
	}
	originalTP := pos.TakeProfitOrders[0].Price
	if originalTP <= 0 {
		return nil
	}

	levels := 5
	if len(ob.Bids) < levels {
		levels = len(ob.Bids)
	}
	if len(ob.Asks) < levels {
		levels = len(ob.Asks)
	}
	if levels == 0 {
		return nil
	}

	var bidDepth, askDepth float64
	for i := 0; i < levels; i++ {
		if i < len(ob.Bids) {
			if v, err := strconv.ParseFloat(ob.Bids[i].Size, 64); err == nil {
				bidDepth += v
			}
		}
		if i < len(ob.Asks) {
			if v, err := strconv.ParseFloat(ob.Asks[i].Size, 64); err == nil {
				askDepth += v
			}
		}
	}

	wallRatio := 0.0
	if pos.Direction == "LONG" && askDepth > 0 {
		wallRatio = bidDepth / askDepth
	} else if pos.Direction == "SHORT" && bidDepth > 0 {
		wallRatio = askDepth / bidDepth
	}

	if wallRatio < densityWallThreshold {
		return nil
	}

	if pos.Direction == "LONG" && currentPrice < originalTP {
		newTP := alignToTick(currentPrice*(1+densityTPPushPct), tickSize)
		if newTP > 0 && math.Abs(newTP-originalTP)/originalTP > 0.001 {
			return &DensityAction{
				Type:    "adjust_tp",
				Symbol:  pos.Symbol,
				TPPrice: newTP,
				Reason:  "massive_ask_wall",
			}
		}
	} else if pos.Direction == "SHORT" && currentPrice > originalTP {
		newTP := alignToTick(currentPrice*(1-densityTPPushPct), tickSize)
		if newTP > 0 && math.Abs(newTP-originalTP)/originalTP > 0.001 {
			return &DensityAction{
				Type:    "adjust_tp",
				Symbol:  pos.Symbol,
				TPPrice: newTP,
				Reason:  "massive_bid_wall",
			}
		}
	}

	return nil
}

func (d *DensityExitManager) checkVelocityReversal(pos *models.ActivePosition) *DensityAction {
	momentum := d.obi.Momentum(pos.Symbol)
	shift := d.obi.PressureShift(pos.Symbol, 0.3)

	if pos.Direction == "LONG" && shift == -1 && momentum < -densityVelocityThresh {
		return &DensityAction{
			Type:   "velocity_reversal_exit",
			Symbol: pos.Symbol,
			Reason: "selling_pressure_reversal",
		}
	}
	if pos.Direction == "SHORT" && shift == 1 && momentum > densityVelocityThresh {
		return &DensityAction{
			Type:   "velocity_reversal_exit",
			Symbol: pos.Symbol,
			Reason: "buying_pressure_reversal",
		}
	}

	return nil
}

func (d *DensityExitManager) checkStagnation(pos *models.ActivePosition, currentPrice float64) *DensityAction {
	holdSec := (time.Now().UnixMilli() - pos.EntryTime) / 1000
	if holdSec < densityStallSec {
		return nil
	}
	if pos.TimeStopPlaced {
		return nil
	}

	rMultiple := 0.0
	if pos.OriginalRisk > 0 {
		if pos.Direction == "LONG" {
			rMultiple = (currentPrice - pos.FillPrice) / pos.OriginalRisk
		} else {
			rMultiple = (pos.FillPrice - currentPrice) / pos.OriginalRisk
		}
	}

	if rMultiple < densityStallR {
		d.logger.Info("[DENSITY INTEL] Stagnation detected — forcing breakeven",
			"symbol", pos.Symbol,
			"hold_sec", holdSec,
			"current_r", fmtF64(rMultiple),
		)
		return &DensityAction{
			Type:   "stagnation_breakeven",
			Symbol: pos.Symbol,
			Reason: "stagnation_breakeven",
		}
	}

	return nil
}

func alignToTick(price float64, tickSize float64) float64 {
	if tickSize <= 0 {
		return price
	}
	return math.Round(price/tickSize) * tickSize
}

func fmtF64(f float64) string {
	s := strconv.FormatFloat(f, 'f', 3, 64)
	return s
}
