package engine

//

import (
	"github.com/mumax/3/cuda"
	"github.com/mumax/3/data"
	"github.com/mumax/3/util"
)

type cropped struct {
	parent                 Quantity
	name                   string
	x1, x2, y1, y2, z1, z2 int
}

func Crop(parent Quantity, name string, x1, x2, y1, y2, z1, z2 int) *cropped {
	n := parent.Mesh().Size()
	util.Argument(x1 < x2 && y1 < y2 && z1 < z2)
	util.Argument(x1 >= 0 && y1 >= 0 && z1 >= 0)
	util.Argument(x2 <= n[X] && y2 <= n[Y] && z2 <= n[Z])
	return &cropped{parent, name, x1, x2, y1, y2, z1, z2}
}

func (q *cropped) NComp() int   { return q.parent.NComp() }
func (q *cropped) Name() string { return q.name }
func (q *cropped) Unit() string { return q.parent.Unit() }

func (q *cropped) Mesh() *data.Mesh {
	c := q.Mesh().CellSize()
	return data.NewMesh(q.x2-q.x1, q.y2-q.y1, q.z2-q.z1, c[X], c[Y], c[Z])
}

//func (q *cropped) average() []float64    { return sAverageUniverse( }
//func (q *cropped) Average() float64      { return q.average()[0] }
//func (q *cropped) Region(r int) *sOneReg { return sOneRegion(q, r) }

func (q *cropped) Slice() (*data.Slice, bool) {
	src, r := q.parent.Slice()
	if r {
		defer cuda.Recycle(src)
	}
	dst := cuda.Buffer(q.NComp(), q.Mesh().Size())
	cuda.Crop(dst, src, q.x1, q.y1, q.z1)
	return dst, true
}
