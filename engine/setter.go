package engine

import (
	"github.com/mumax/3/cuda"
	"github.com/mumax/3/data"
)

// quantity that is not stored, but can output to (set) a buffer
type _setter struct {
	_set func(dst *data.Slice) // calculates quantity and stores in dst
	info
}

// initialize setter and declare as quantity (for script and gui)
func (q *_setter) _init(nComp int, name, unit, doc string, setFunc func(dst *data.Slice)) {
	*q = _setter{setFunc, Info(nComp, name, unit)}
}

// get the quantity, recycle will be true (q needs to be recycled)
func (b *_setter) Slice() (q *data.Slice, recycle bool) {
	buffer := cuda.Buffer(b.NComp(), b.Mesh().Size())
	b.Set(buffer)
	return buffer, true // must recycle
}

func (q *_setter) average() []float64  { return qAverageUniverse(q) }
func (q *_setter) Set(dst *data.Slice) { q._set(dst) }
func (q *_setter) Mesh() *data.Mesh    { return Mesh() }

type sSetter struct{ _setter }

func (q *sSetter) init(name, unit, doc string, setFunc func(dst *data.Slice)) {
	q._init(1, name, unit, doc, setFunc)
	DeclROnly(name, q, cat(doc, unit))
}
func (q *sSetter) Average() float64      { return q.average()[0] }
func (q *sSetter) Region(r int) *sOneReg { return sOneRegion(q, r) }

type vSetter struct{ _setter }

func (q *vSetter) init(name, unit, doc string, setFunc func(dst *data.Slice)) {
	q._init(3, name, unit, doc, setFunc)
	DeclROnly(name, q, cat(doc, unit))
}
func (q *vSetter) Average() data.Vector  { return unslice(q.average()) }
func (q *vSetter) Region(r int) *vOneReg { return vOneRegion(q, r) }
func (q *vSetter) Comp(c int) *comp      { return Comp(q, c) }
