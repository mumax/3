package engine

// Add arbitrary terms to B_eff, Edens_total.

import (
	"github.com/mumax/3/cuda"
	"github.com/mumax/3/data"
)

var (
	PosLocal = NewVectorField("PositionLocal", "None", "Vector field containing position of each cell normalised from -1 to +1 across each dimension.", PositionLocal)
	PosWorld = NewVectorField("PositionWorld", "m", "Vector field containing real world position of each cell. The Origin is at the center of the mesh", PositionWorld)
)

func PositionLocal(dst *data.Slice) {

	dims := Mesh().Size()

	sx := 1.0 / float32(dims[X])
	sy := 1.0 / float32(dims[Y])
	sz := 1.0 / float32(dims[Z])

	cuda.SpatialField(dst, sx, sy, sz)
}

func PositionWorld(dst *data.Slice) {

	dims := Mesh().Size()
	cellsize := Mesh().CellSize()

	sx := float32(cellsize[X]) / float32(dims[X])
	sy := float32(cellsize[Y]) / float32(dims[Y])
	sz := float32(cellsize[Z]) / float32(dims[Z])

	cuda.SpatialField(dst, sx, sy, sz)
}
