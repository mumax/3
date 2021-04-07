package engine

// Add arbitrary terms to B_eff, Edens_total.

import (
	"github.com/mumax/3/cuda"
	"github.com/mumax/3/data"
)

var (
	PosLocal = NewVectorField("PositionLocal", "None", "Vector field containing position of each cell normalised from -0.5 to +0.5 across each dimension.", PositionLocal)
	PosWorld = NewVectorField("PositionWorld", "m", "Vector field containing real world position of each cell. The Origin is at the center of the mesh", PositionWorld)
)

func decrement_safe(x int) int {
	if x > 1 {
		x = x - 1
	}

	return x
}

func PositionLocal(dst *data.Slice) {

	dims := Mesh().Size()

	sx := 1.0 / float32(decrement_safe(dims[X]))
	sy := 1.0 / float32(decrement_safe(dims[Y]))
	sz := 1.0 / float32(decrement_safe(dims[Z]))

	cuda.SpatialField(dst, sx, sy, sz)
}

func PositionWorld(dst *data.Slice) {

	cellsize := Mesh().CellSize()

	sx := float32(cellsize[X])
	sy := float32(cellsize[Y])
	sz := float32(cellsize[Z])

	cuda.SpatialField(dst, sx, sy, sz)
}
