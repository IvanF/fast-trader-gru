package lineproto

import (
	"fmt"
	"strconv"
	"strings"
	"time"

	"github.com/fast-trader-gru/history_logger/internal/models"
)

func escapeTag(s string) string {
	return strings.NewReplacer(
		",", `\,`,
		"=", `\=`,
		" ", `\ `,
	).Replace(s)
}

func escapeStringField(s string) string {
	return strings.ReplaceAll(s, `"`, `\"`)
}

// TradeLine returns a Line Protocol point for a public trade tick.
func TradeLine(t models.TradePayload) string {
	tsNano := t.Ts * int64(time.Millisecond)
	sym := escapeTag(t.Symbol)
	side := escapeStringField(t.Side)
	return fmt.Sprintf(
		"trades,symbol=%s side=\"%s\",price=%s,size=%s,trade_id=\"%s\" %d",
		sym, side,
		formatFloat(t.Price),
		formatFloat(t.Size),
		escapeStringField(t.TradeID),
		tsNano,
	)
}

// OrderbookDepthLines emits one point per bid/ask level up to depth.
func OrderbookDepthLines(ob models.OrderbookPayload, depth int) []string {
	if depth <= 0 {
		depth = 50
	}
	tsNano := ob.Ts * int64(time.Millisecond)
	sym := escapeTag(ob.Symbol)
	lines := make([]string, 0, len(ob.Bids)+len(ob.Asks)+1)

	var bidVol, askVol float64
	for i, b := range ob.Bids {
		if i >= depth {
			break
		}
		p, _ := strconv.ParseFloat(b.Price, 64)
		s, _ := strconv.ParseFloat(b.Size, 64)
		bidVol += s
		lines = append(lines, fmt.Sprintf(
			"orderbook_depth,symbol=%s,side=bid level=%di,price=%s,size=%s %d",
			sym, i, formatFloat(p), formatFloat(s), tsNano,
		))
	}
	for i, a := range ob.Asks {
		if i >= depth {
			break
		}
		p, _ := strconv.ParseFloat(a.Price, 64)
		s, _ := strconv.ParseFloat(a.Size, 64)
		askVol += s
		lines = append(lines, fmt.Sprintf(
			"orderbook_depth,symbol=%s,side=ask level=%di,price=%s,size=%s %d",
			sym, i, formatFloat(p), formatFloat(s), tsNano,
		))
	}

	total := bidVol + askVol
	obi := 0.0
	if total > 0 {
		obi = (bidVol - askVol) / total
	}
	lines = append(lines, fmt.Sprintf(
		"orderbook_summary,symbol=%s bid_vol=%s,ask_vol=%s,obi=%s,levels=%di %d",
		sym, formatFloat(bidVol), formatFloat(askVol), formatFloat(obi), depth, tsNano,
	))
	return lines
}

func formatFloat(v float64) string {
	return strconv.FormatFloat(v, 'f', -1, 64)
}
