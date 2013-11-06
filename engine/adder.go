package engine

import (
	"github.com/mumax/3/cuda"
	"github.com/mumax/3/data"
)

// quantity that is not explicitly stored,
// but only added to an other quantity (like effective field contributions)
type adder struct {
	addTo func(dst *data.Slice) // calculates quantity and add result to dst
	info
}

func (q *adder) init(nComp int, m *data.Mesh, name, unit, doc string, addFunc func(dst *data.Slice)) {
	*q = adder{addFunc, Info(nComp, name, unit, m)}
	DeclROnly(name, q, cat(doc, unit))
}

// Calcuates and returns the quantity.
// recycle is true: slice needs to be recycled.
func (q *adder) Slice() (s *data.Slice, recycle bool) {
	buf := cuda.Buffer(q.NComp(), q.Mesh())
	cuda.Zero(buf)
	q.addTo(buf)
	return buf, true
}

// Output for data table.
func (q *adder) TableData() []float64 { return Average(q) }

// Value of this quantity restricted to one region.
func (q *adder) Region(r int) *sliceInRegion { return &sliceInRegion{q, r} }
