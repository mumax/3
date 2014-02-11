package engine

// Comp is a Derived Quantity pointing to a single component of vector Quantity

import (
	"fmt"
	"github.com/mumax/3/cuda"
	"github.com/mumax/3/data"
	"github.com/mumax/3/util"
)

type comp struct {
	parent Quantity
	comp   int
}

// Comp returns vector component c of the parent Quantity
func Comp(parent Quantity, c int) *comp {
	util.Argument(c >= 0 && c < parent.NComp())
	return &comp{parent, c}
}

func (q *comp) NComp() int            { return 1 }
func (q *comp) Name() string          { return fmt.Sprint(q.parent.Name(), "_", compname[q.comp]) }
func (q *comp) Unit() string          { return q.parent.Unit() }
func (q *comp) Mesh() *data.Mesh      { return q.parent.Mesh() }
func (q *comp) average() []float64    { return []float64{q.parent.average()[q.comp]} }
func (q *comp) Average() float64      { return q.average()[0] }
func (q *comp) Region(r int) *sOneReg { return sOneRegion(q, r) }

func (q *comp) Slice() (*data.Slice, bool) {
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
