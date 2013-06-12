package engine

import (
	"code.google.com/p/mx3/cuda"
	"code.google.com/p/mx3/data"
	"code.google.com/p/mx3/util"
)

// TODO: remove?

type maskQuant struct {
	bufferedQuant
}

func mask(nComp int, m *data.Mesh, name, unit string) maskQuant {
	slice := data.NilSlice(nComp, m)
	return maskQuant{buffered(slice, name, unit)}
}

// Set the value of all cell faces with their normal along direction. E.g.:
// 	SetAll(X, 1) // sets all faces in YZ plane to value 1.
func (m *maskQuant) SetAll(component int, value float64) {
	m.alloc()
	cuda.Memset(m.buffer.Comp(util.SwapIndex(component, 3)), float32(value))
}
