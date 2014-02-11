package engine

import (
	"github.com/mumax/3/cuda"
	"github.com/mumax/3/data"
)

// quantity that is not explicitly stored,
// but only added to an other quantity (like effective field contributions)
type _adder struct {
	_addTo func(dst *data.Slice) // calculates quantity and add result to dst
	info
}

func (q *_adder) _init(nComp int, name, unit, doc string, addFunc func(dst *data.Slice)) {
	*q = _adder{addFunc, Info(nComp, name, unit)}
}

// Calculates and returns the quantity.
// recycle is true: slice needs to be recycled.
func (q *_adder) Slice() (s *data.Slice, recycle bool) {
	buf := cuda.Buffer(q.NComp(), q.Mesh().Size())
	q.Set(buf)
	return buf, true
}

// Output for data table.
func (q *_adder) average() []float64    { return qAverageUniverse(q) }
func (q *_adder) Mesh() *data.Mesh      { return Mesh() }
func (q *_adder) AddTo(dst *data.Slice) { q._addTo(dst) }

func (q *_adder) Set(dst *data.Slice) {
	cuda.Zero(dst)
	q.AddTo(dst)
}

// scalar adder
type sAdder struct{ _adder }

func (q *sAdder) init(name, unit, doc string, addFunc func(dst *data.Slice)) {
	q._init(1, name, unit, doc, addFunc)
	DeclROnly(name, q, cat(doc, unit))
}
func (q *sAdder) Average() float64      { return q.average()[0] }
func (q *sAdder) Region(r int) *sOneReg { return sOneRegion(q, r) }

// vector adder
type vAdder struct{ _adder }

func (q *vAdder) init(name, unit, doc string, addFunc func(dst *data.Slice)) {
	q._init(3, name, unit, doc, addFunc)
	DeclROnly(name, q, cat(doc, unit))
}
func (q *vAdder) Average() data.Vector  { return unslice(q.average()) }
func (q *vAdder) Region(r int) *vOneReg { return vOneRegion(q, r) }
func (q *vAdder) Comp(c int) *comp      { return Comp(q, c) }
