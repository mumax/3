package engine

// Comp is a Derived Quantity pointing to a single component of vector Quantity

import (
	"fmt"
	"github.com/mumax/3/cuda"
	"github.com/mumax/3/data"
	"github.com/mumax/3/util"
)

type component struct {
	parent outputField
	comp   int
}

// Comp returns vector component c of the parent Quantity
func Comp(parent outputField, c int) ScalarField {
	util.Argument(c >= 0 && c < parent.NComp())
	return AsScalarField(&component{parent, c})
}

func (q *component) NComp() int         { return 1 }
func (q *component) Name() string       { return fmt.Sprint(q.parent.Name(), "_", compname[q.comp]) }
func (q *component) Unit() string       { return q.parent.Unit() }
func (q *component) Mesh() *data.Mesh   { return q.parent.Mesh() }
func (q *component) average() []float64 { return []float64{q.parent.average()[q.comp]} } // TODO

func (q *component) Slice() (*data.Slice, bool) {
	p := q.parent
	src, r := p.Slice()
	if r {
		for i := 0; i < p.NComp(); i++ {
			if i != q.comp {
				defer cuda.Recycle(src.Comp(i))
			}
		}
	}
	return src.Comp(q.comp), r
}

var compname = map[int]string{0: "x", 1: "y", 2: "z"}
