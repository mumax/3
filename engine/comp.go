package engine

// Comp is a Derived Quantity pointing to a single component of vector Quantity

import (
	"fmt"
	"github.com/mumax/3/v3/cuda"
	"github.com/mumax/3/v3/data"
	"github.com/mumax/3/v3/util"
)

type component struct {
	parent Quantity
	comp   int
}

// Comp returns vector component c of the parent Quantity
func Comp(parent Quantity, c int) ScalarField {
	util.Argument(c >= 0 && c < parent.NComp())
	return AsScalarField(&component{parent, c})
}

func (q *component) NComp() int       { return 1 }
func (q *component) Name() string     { return fmt.Sprint(NameOf(q.parent), "_", compname[q.comp]) }
func (q *component) Unit() string     { return UnitOf(q.parent) }
func (q *component) Mesh() *data.Mesh { return MeshOf(q.parent) }

func (q *component) Slice() (*data.Slice, bool) {
	p := q.parent
	src := ValueOf(p)
	defer cuda.Recycle(src)
	c := cuda.Buffer(1, src.Size())
	return c, true
}

func (q *component) EvalTo(dst *data.Slice) {
	src := ValueOf(q.parent)
	defer cuda.Recycle(src)
	data.Copy(dst, src.Comp(q.comp))
}

var compname = map[int]string{0: "x", 1: "y", 2: "z"}
