package engine

import (
	"github.com/mumax/3/cuda"
	"github.com/mumax/3/data"
)

// quantity that is not explicitly stored,
// but only added to an other quantity (like effective field contributions)
type adder struct {
	_addTo func(dst *data.Slice) // calculates quantity and add result to dst
	info
}

func (q *adder) init(nComp int, name, unit, doc string, addFunc func(dst *data.Slice)) {
	*q = adder{addFunc, Info(nComp, name, unit)}
	DeclROnly(name, q, cat(doc, unit))
}

// Calculates and returns the quantity.
// recycle is true: slice needs to be recycled.
func (q *adder) Slice() (s *data.Slice, recycle bool) {
	buf := cuda.Buffer(q.NComp(), q.Mesh().Size())
	q.Set(buf)
	return buf, true
}

// Output for data table.
func (q *adder) TableData() []float64 { return qAverageUniverse(q) }

func (q *adder) Mesh() *data.Mesh { return Mesh() }

// Value of this quantity restricted to one region.
func (q *adder) Region(r int) *sliceInRegion { return &sliceInRegion{q, r} }

func (q *adder) AddTo(dst *data.Slice) { q._addTo(dst) }

func (q *adder) Set(dst *data.Slice) {
	cuda.Zero(dst)
	q.AddTo(dst)
}

//func (q *adder) Comp(c int) *comp { return Comp(q, c) }
