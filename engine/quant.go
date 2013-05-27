package engine

import (
	"code.google.com/p/mx3/data"
)

type Quant interface {
	Download() *data.Slice
}
