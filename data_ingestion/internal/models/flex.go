package models

import (
	"encoding/json"
	"fmt"
	"strconv"
)

// FlexInt64 accepts JSON numbers or numeric strings (Bybit trade timestamp field T).
type FlexInt64 int64

func (f *FlexInt64) UnmarshalJSON(data []byte) error {
	var v any
	if err := json.Unmarshal(data, &v); err != nil {
		return err
	}
	switch x := v.(type) {
	case float64:
		*f = FlexInt64(int64(x))
		return nil
	case string:
		n, err := strconv.ParseInt(x, 10, 64)
		if err != nil {
			return err
		}
		*f = FlexInt64(n)
		return nil
	default:
		return fmt.Errorf("flex int64: unexpected type %T", v)
	}
}
