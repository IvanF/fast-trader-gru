package grid

import (
	"strconv"

	"github.com/fast-trader-gru/oms_execution/internal/models"
)

func MidPrice(ob models.OrderbookSnapshot) float64 {
	if len(ob.Bids) == 0 || len(ob.Asks) == 0 {
		return 0
	}
	bid, _ := strconv.ParseFloat(ob.Bids[0].Price, 64)
	ask, _ := strconv.ParseFloat(ob.Asks[0].Price, 64)
	if bid == 0 || ask == 0 {
		return 0
	}
	return (bid + ask) / 2
}
