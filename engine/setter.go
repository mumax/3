package engine

import (
	"github.com/mumax/3/cuda"
	"github.com/mumax/3/data"
)

// quantity that is not stored, but can output to (set) a buffer
type setter struct {
	_set func(dst *data.Slice) // calculates quantity and stores in dst
	info
}

// initialize setter and declare as quantity (for script and gui)
func (q *setter) init(nComp int, name, unit, doc string, setFunc func(dst *data.Slice)) {
	*q = setter{setFunc, Info(nComp, name, unit)}
	DeclROnly(name, q, cat(doc, unit))
}

// get the quantity, recycle will be true (q needs to be recycled)
func (b *setter) Slice() (q *data.Slice, recycle bool) {
	buffer := cuda.Buffer(b.NComp(), b.Mesh().Size())
	b.Set(buffer)
	return buffer, true // must recycle
}

func (q *setter) TableData() []float64    { return qAverageUniverse(q) }
func (q *setter) Region(r int) *oneRegion { return &oneRegion{q, r} }
func (q *setter) Set(dst *data.Slice)     { q._set(dst) }
func (q *setter) Mesh() *data.Mesh        { return Mesh() }

//func (q *setter) Comp(c int) *comp            { return Comp(q, c) }
