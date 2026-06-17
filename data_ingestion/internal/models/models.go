package models

import "encoding/json"

type ActiveSymbolsPayload struct {
	UpdatedAt int64    `json:"updated_at" msgpack:"updated_at"`
	Symbols   []string `json:"symbols" msgpack:"symbols"`
}

// OrderbookLevel accepts Bybit v5 [price, size] arrays and {price, size} objects.
type OrderbookLevel struct {
	Price string `json:"price" msgpack:"price"`
	Size  string `json:"size" msgpack:"size"`
}

func (l *OrderbookLevel) UnmarshalJSON(data []byte) error {
	var arr []string
	if err := json.Unmarshal(data, &arr); err == nil && len(arr) >= 2 {
		l.Price, l.Size = arr[0], arr[1]
		return nil
	}
	var obj struct {
		Price string `json:"price"`
		Size  string `json:"size"`
	}
	if err := json.Unmarshal(data, &obj); err != nil {
		return err
	}
	l.Price, l.Size = obj.Price, obj.Size
	return nil
}

type OrderbookPayload struct {
	Symbol   string           `json:"symbol" msgpack:"symbol"`
	Ts       int64            `json:"ts" msgpack:"ts"`
	Bids     []OrderbookLevel `json:"b" msgpack:"b"`
	Asks     []OrderbookLevel `json:"a" msgpack:"a"`
	UpdateID uint64           `json:"u" msgpack:"u"`
	Seq      uint64           `json:"seq" msgpack:"seq"`
}

type TradePayload struct {
	Symbol  string  `json:"symbol" msgpack:"symbol"`
	Ts      int64   `json:"ts" msgpack:"ts"`
	Price   float64 `json:"p" msgpack:"p"`
	Size    float64 `json:"v" msgpack:"v"`
	Side    string  `json:"S" msgpack:"S"`
	TradeID string  `json:"i" msgpack:"i"`
	IsBlock bool    `json:"block" msgpack:"block"`
}
