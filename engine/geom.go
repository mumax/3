package engine

import (
	"code.google.com/p/mx3/cuda"
	"code.google.com/p/mx3/data"
	"log"
)

var (
	vol       *data.Slice // cell fillings (0..1)
	spaceFill = 1.0       // filled fraction of space
)

func init() {
	DeclFunc("SetGeom", SetGeom, "Sets the geometry to a given shape")
}

func SetGeom(s Shape) {
	if vol.IsNil() {
		vol = cuda.NewSlice(1, Mesh())
	}
	V := data.NewSlice(1, vol.Mesh())
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

	spaceFill = fill / float64(Mesh().NCell())
	if spaceFill == 0 {
		log.Fatal("SetGeom: geometry completely empty")
	}

	data.Copy(vol, V)
	cuda.Normalize(M.buffer, vol) // removes m outside vol
}
