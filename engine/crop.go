package engine

// Cropped quantity refers to a cut-out piece of a large quantity

import (
	"fmt"
	"github.com/mumax/3/cuda"
	"github.com/mumax/3/data"
	"github.com/mumax/3/util"
)

func init() {
	DeclFunc("Crop", Crop, "Crops a quantity to cell ranges [x1,x2[, [y1,y2[, [z1,z2[")
	DeclFunc("CropX", CropX, "Crops a quantity to cell ranges [x1,x2[")
	DeclFunc("CropY", CropY, "Crops a quantity to cell ranges [y1,y2[")
	DeclFunc("CropZ", CropZ, "Crops a quantity to cell ranges [z1,z2[")
	DeclFunc("CropLayer", CropLayer, "Crops a quantity to a single layer")
}

type cropped struct {
	parent                 Quantity
	name                   string
	x1, x2, y1, y2, z1, z2 int
}

func CropLayer(parent Quantity, layer int) *cropped {
	n := parent.Mesh().Size()
	return Crop(parent, 0, n[X], 0, n[Y], layer, layer+1)
}

func CropX(parent Quantity, x1, x2 int) *cropped {
	n := parent.Mesh().Size()
	return Crop(parent, x1, x2, 0, n[Y], 0, n[Z])
}

func CropY(parent Quantity, y1, y2 int) *cropped {
	n := parent.Mesh().Size()
	return Crop(parent, 0, n[X], y1, y2, 0, n[Z])
}

func CropZ(parent Quantity, z1, z2 int) *cropped {
	n := parent.Mesh().Size()
	return Crop(parent, 0, n[X], 0, n[Y], z1, z2)
}

func Crop(parent Quantity, x1, x2, y1, y2, z1, z2 int) *cropped {
	n := parent.Mesh().Size()
	util.Argument(x1 < x2 && y1 < y2 && z1 < z2)
	util.Argument(x1 >= 0 && y1 >= 0 && z1 >= 0)
	util.Argument(x2 <= n[X] && y2 <= n[Y] && z2 <= n[Z])

	name := parent.Name()
	if x1 != 0 || x2 != n[X] {
		name += "_xrange" + rangeStr(x1, x2)
	}
	if y1 != 0 || y2 != n[Y] {
		name += "_yrange" + rangeStr(y1, y2)
	}
	if z1 != 0 || z2 != n[Z] {
		name += "_zrange" + rangeStr(z1, z2)
	}

	return &cropped{parent, name, x1, x2, y1, y2, z1, z2}
}

func rangeStr(a, b int) string {
	if a+1 == b {
		return fmt.Sprint(a, "_")
	} else {
		return fmt.Sprint(a, "-", b, "_")
	}
	// (trailing underscore to separate from subsequent autosave number)
}

func (q *cropped) NComp() int   { return q.parent.NComp() }
func (q *cropped) Name() string { return q.name }
func (q *cropped) Unit() string { return q.parent.Unit() }

func (q *cropped) Mesh() *data.Mesh {
	c := q.parent.Mesh().CellSize()
	return data.NewMesh(q.x2-q.x1, q.y2-q.y1, q.z2-q.z1, c[X], c[Y], c[Z])
}

func (q *cropped) average() []float64 { return qAverageUniverse(q) } // needed for table
func (q *cropped) Average() []float64 { return q.average() }         // handy for script

func (q *cropped) Slice() (*data.Slice, bool) {
	src, r := q.parent.Slice()
	if r {
		defer cuda.Recycle(src)
	}
	dst := cuda.Buffer(q.NComp(), q.Mesh().Size())
	cuda.Crop(dst, src, q.x1, q.y1, q.z1)
	return dst, true
}
