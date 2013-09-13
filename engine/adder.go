package engine

import (
	"code.google.com/p/mx3/cuda"
	"code.google.com/p/mx3/data"
)

// quantity that is not explicitly stored,
// but only added to an other quantity (like effective field contributions)
type adder struct {
	addTo func(dst *data.Slice) // calculates quantity and add result to dst
	info
}

func (q *adder) init(nComp int, m *data.Mesh, name, unit, doc string, addFunc func(dst *data.Slice)) {
	*q = adder{addFunc, Info(nComp, name, unit, m)}
	DeclROnly(name, q, doc)
}

// Calcuates and returns the quantity.
// recycle is true: slice needs to be recycled.
func (a *adder) Get() (q *data.Slice, recycle bool) {
	buf := cuda.GetBuffer(a.NComp(), a.Mesh())
	cuda.Zero(buf)
	a.addTo(buf)
	return buf, true
}
