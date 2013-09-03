package engine

import (
	"code.google.com/p/mx3/cuda"
	"code.google.com/p/mx3/data"
)

// quantity that is not stored, but can output to (set) a buffer
type setterQuant struct {
	set func(dst *data.Slice) // calculates quantity and stores in dst
	info
}

// constructor
func setter(nComp int, m *data.Mesh, name, unit string, setFunc func(dst *data.Slice)) setterQuant {
	return setterQuant{setFunc, info{nComp, name, unit, m}}
}

// get the quantity, recycle will be true (q needs to be recycled)
func (b *setterQuant) Get() (q *data.Slice, recycle bool) {
	buffer := cuda.GetBuffer(b.nComp, b.mesh)
	b.set(buffer)
	return buffer, true // must recycle
}
