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
	info
	buffer *data.Slice
	shape  Shape
}

func (g *geom) init() {
	g.buffer = nil
	g.info = Info(1, "geom", "")
	DeclROnly("geom", &geometry, "Cell fill fraction (0..1)")
}

func spaceFill() float64 {
	if geometry.Gpu().IsNil() {
		return 1
	} else {
		return float64(cuda.Sum(geometry.buffer)) / float64(geometry.Mesh().NCell())
	}
}

func (g *geom) Gpu() *data.Slice {
	if g.buffer == nil {
		g.buffer = data.NilSlice(1, g.Mesh().Size())
	}
	return g.buffer
}

func SetGeom(s Shape) {
	geometry.setGeom(s)
}

func (geometry *geom) setGeom(s Shape) {
	geometry.shape = s
	if geometry.Gpu().IsNil() {
		geometry.buffer = cuda.NewSlice(1, geometry.Mesh().Size())
	}
	if s == nil {
		s = universe
	}

	host := data.NewSlice(1, geometry.Gpu().Size())
	array := host.Scalars()
	V := host
	v := array
	n := Mesh().Size()

	var ok bool
	for iz := 0; iz < n[Z]; iz++ {
		for iy := 0; iy < n[Y]; iy++ {
			for ix := 0; ix < n[X]; ix++ {
				r := Index2Coord(ix, iy, iz)
				x, y, z := r[X], r[Y], r[Z]
				if s(x, y, z) { // inside
					v[iz][iy][ix] = 1
					ok = true
				} else {
					v[iz][iy][ix] = 0
				}
			}
		}
	}

	if !ok {
		util.Fatal("SetGeom: geometry completely empty")
	}

	data.Copy(geometry.buffer, V)
	cuda.Normalize(M.Buffer(), geometry.Gpu()) // removes m outside vol
}

func (g *geom) shift(dx int) {
	// empty mask, nothing to do
	if g.buffer.IsNil() {
		return
	}

	// allocated mask: shift
	s := g.buffer
	s2 := cuda.Buffer(1, g.Mesh().Size())
	defer cuda.Recycle(s2)
	newv := float32(1) // initially fill edges with 1's
	cuda.ShiftX(s2, s, dx, newv, newv)
	data.Copy(s, s2)

	n := Mesh().Size()
	nx := n[X]

	// re-evaluate edge regions
	var x1, x2 int
	util.Argument(dx != 0)
	if dx < 0 {
		x1 = nx + dx
		x2 = nx
	} else {
		x1 = 0
		x2 = dx
	}

	for iz := 0; iz < n[Z]; iz++ {
		for iy := 0; iy < n[Y]; iy++ {
			for ix := x1; ix < x2; ix++ {
				r := Index2Coord(ix, iy, iz) // includes shift
				if !g.shape(r[X], r[Y], r[Z]) {
					cuda.SetCell(g.buffer, 0, ix, iy, iz, 0) // a bit slowish, but hardly reached
				}
			}
		}
	}

	cuda.Normalize(M.Buffer(), geometry.Gpu())
}

func (g *geom) Mesh() *data.Mesh { return Mesh() }
