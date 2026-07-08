package executor

import (
	"context"
	"fmt"
	"log/slog"
	"math"
	"strconv"
	"time"

	"github.com/fast-trader-gru/oms_execution/internal/bybit"
	"github.com/fast-trader-gru/oms_execution/internal/models"
)

const (
	exitChaserMaxAttempts = 5
	exitChaserIntervalSec = 3
	exitChaserRepriceTick = 1
)

// ExitChaser provides passive limit exit chasing to avoid Taker fees.
// Instead of market Kill-Switch, it chases the spread with PostOnly limits.
type ExitChaser struct {
	logger   *slog.Logger
	bybit    *bybit.Client
}

func NewExitChaser(logger *slog.Logger, client *bybit.Client) *ExitChaser {
	return &ExitChaser{
		logger: logger,
		bybit:  client,
	}
}

// ExecutePassiveExit runs a limit-chasing loop to close position as Maker.
// Returns (finalPrice, error). On full failure, the caller should fall back to market.
func (ec *ExitChaser) ExecutePassiveExit(
	ctx context.Context,
	pos *models.ActivePosition,
	ob models.OrderbookSnapshot,
	tickSize float64,
) (float64, error) {

	side := "Sell"
	if pos.Direction != "LONG" {
		side = "Buy"
	}

	ec.logger.Info("[EXIT-CHASER] Starting passive limit exit",
		"symbol", pos.Symbol, "direction", pos.Direction, "qty", pos.RemainingQty)

	for attempt := 1; attempt <= exitChaserMaxAttempts; attempt++ {
		if ctx.Err() != nil {
			return 0, ctx.Err()
		}

		// Check if position is still open
		exPos, exErr := ec.bybit.GetPosition(ctx, pos.Symbol)
		if exErr == nil && exPos.Size <= 0 {
			ec.logger.Info("[EXIT-CHASER] Position already closed on exchange",
				"symbol", pos.Symbol)
			return 0, nil
		}

		// Calculate target price inside the spread (Maker-safe)
		targetPrice := ec.calculateExitPrice(pos.Direction, ob, tickSize)
		if targetPrice <= 0 {
			ec.logger.Warn("[EXIT-CHASER] Cannot calculate exit price, skipping attempt",
				"symbol", pos.Symbol, "attempt", attempt)
			time.Sleep(1 * time.Second)
			continue
		}

		qty := pos.RemainingQty
		if qty < pos.MinOrderQty {
			qty = pos.MinOrderQty
		}

		ec.logger.Info("[EXIT-CHASER] Placing PostOnly limit exit",
			"symbol", pos.Symbol, "attempt", attempt,
			"price", targetPrice, "qty", qty, "side", side)

		orderID, orderErr := ec.bybit.PlaceReducePostOnlyLimit(ctx, pos.Symbol, side, qty, pos.QtyStep, bybit.FormatPrice(targetPrice))
		if orderErr != nil {
			ec.logger.Warn("[EXIT-CHASER] PostOnly rejected, retrying",
				"symbol", pos.Symbol, "error", orderErr)
			time.Sleep(1 * time.Second)
			continue
		}

		// Wait for fill
		time.Sleep(time.Duration(exitChaserIntervalSec) * time.Second)

		// Check fill status
		oi, oiErr := ec.bybit.GetOrderRealtime(ctx, pos.Symbol, orderID)
		if oiErr == nil && oi.OrderStatus == "Filled" {
			ec.logger.Info("[EXIT-CHASER] Exit filled as Maker",
				"symbol", pos.Symbol, "price", targetPrice, "attempt", attempt)
			return targetPrice, nil
		}

		// Cancel unfilled order
		_ = ec.bybit.CancelOrder(ctx, pos.Symbol, orderID)

		ec.logger.Info("[EXIT-CHASER] Order not filled, repricing",
			"symbol", pos.Symbol, "attempt", attempt, "status", oi.OrderStatus)
	}

	// All attempts exhausted — return error so caller can fallback
	ec.logger.Warn("[EXIT-CHASER] Passive chasing exhausted after max attempts",
		"symbol", pos.Symbol, "attempts", exitChaserMaxAttempts)
	return 0, fmt.Errorf("exit chaser exhausted: %d attempts failed", exitChaserMaxAttempts)
}

// calculateExitPrice determines the optimal Maker exit price inside the spread.
func (ec *ExitChaser) calculateExitPrice(direction string, ob models.OrderbookSnapshot, tickSize float64) float64 {
	if len(ob.Bids) == 0 || len(ob.Asks) == 0 {
		return 0
	}

	bestBid, _ := strconv.ParseFloat(ob.Bids[0].Price, 64)
	bestAsk, _ := strconv.ParseFloat(ob.Asks[0].Price, 64)

	if bestBid <= 0 || bestAsk <= 0 {
		return 0
	}

	if direction == "LONG" {
		// Selling to exit: place limit at Best Bid + 1 tick (still inside spread)
		target := bestBid + tickSize
		if target >= bestAsk {
			target = bestAsk - tickSize
		}
		return math.Round(target/tickSize) * tickSize
	} else {
		// Buying to exit: place limit at Best Ask - 1 tick (still inside spread)
		target := bestAsk - tickSize
		if target <= bestBid {
			target = bestBid + tickSize
		}
		return math.Round(target/tickSize) * tickSize
	}
}
