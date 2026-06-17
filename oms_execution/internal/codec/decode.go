package codec

import (
	"encoding/json"

	"github.com/vmihailenco/msgpack/v5"
)

func Unmarshal(raw []byte, dest any) error {
	if err := msgpack.Unmarshal(raw, dest); err == nil {
		return nil
	}
	return json.Unmarshal(raw, dest)
}
