package cuda

import (
	"code.google.com/p/mx3/data"
)

func NewQuant(nComp int, mesh *data.Mesh) *data.Quant {
	slice := NewSlice(nComp, mesh)
	return data.QuantFromSlice(slice)
}
