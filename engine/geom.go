package engine

import (
	"github.com/mumax/3/cuda"
	"github.com/mumax/3/data"
	"log"
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
	g.buffered.init(1, "geometry", "", "Cell fill fraction", &globalmesh)
	//DeclROnly("geometry", )
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
	c := Mesh().CellSize()
	dx := (float64(n[2]/2) - 0.5) * c[2]
	dy := (float64(n[1]/2) - 0.5) * c[1]
	dz := (float64(n[0]/2) - 0.5) * c[0]

	fill := 0.0

	for i := 0; i < n[0]; i++ {
		z := float64(i)*c[0] - dz
		for j := 0; j < n[1]; j++ {
			y := float64(j)*c[1] - dy
			for k := 0; k < n[2]; k++ {
				x := float64(k)*c[2] - dx
				if s(x, y, z) { // inside
					v[i][j][k] = 1
					fill += 1.0
				} else {
					v[i][j][k] = 0
				}
			}
		}
	}

	if fill == 0 {
		log.Fatal("SetGeom: geometry completely empty")
	}
	geometry.spaceFill = fill / float64(Mesh().NCell())

	data.Copy(geometry.buffer, V)
	cuda.Normalize(M.buffer, vol()) // removes m outside vol
}

func (g *geom) shift(dx int) {
	if g.buffer.IsNil() {
		return
	}
	panic("TODO: shift geom")
}
