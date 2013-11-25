package engine

import (
	"github.com/mumax/3/cuda"
	"github.com/mumax/3/data"
	"github.com/mumax/3/util"
)

func init() {
	DeclFunc("SetGeom", SetGeom, "Sets the geometry to a given shape")
	geometry.init()
}

var geometry geom

type geom struct {
	buffered          // cell fillings (0..1)
	spaceFill float64 // filled fraction of space
	shape     Shape
}

func (g *geom) init() {
	g.buffered.init(1, "geometry", "", &globalmesh)
	DeclROnly("geometry", &geometry, "Cell fill fraction (0..1)")
	g.spaceFill = 1.0 // filled fraction of space
}

func vol() *data.Slice {
	return geometry.Gpu()
}

func spaceFill() float64 {
	return geometry.spaceFill
}

func (g *geom) Gpu() *data.Slice {
	if g.buffer == nil {
		g.buffer = data.NilSlice(1, Mesh())
	}
	return g.buffer
}

func SetGeom(s Shape) {
	geometry.setGeom(s)
}

func (geometry *geom) setGeom(s Shape) {
	geometry.shape = s
	if vol().IsNil() {
		geometry.buffer = cuda.NewSlice(1, Mesh())
	}
	V := data.NewSlice(1, vol().Mesh())
	v := V.Scalars()
	n := Mesh().Size()

	fill := 0.0

	for iz := 0; iz < n[Z]; iz++ {
		for iy := 0; iy < n[Y]; iy++ {
			for ix := 0; ix < n[X]; ix++ {
				r := Index2Coord(ix, iy, iz)
				x, y, z := r[X], r[Y], r[Z]
				if s(x, y, z) { // inside
					v[iz][iy][ix] = 1
					fill += 1.0
				} else {
					v[iz][iy][ix] = 0
				}
			}
		}
	}

	if fill == 0 {
		util.Fatal("SetGeom: geometry completely empty")
	}
	geometry.spaceFill = fill / float64(Mesh().NCell())

	data.Copy(geometry.buffer, V)
	cuda.Normalize(M.Buffer(), vol()) // removes m outside vol
}

func (g *geom) shift(dx int) {
	if g.buffer.IsNil() {
		return
	}
	panic("todo")
	shiftSlice(g.buffer, dx)
}
