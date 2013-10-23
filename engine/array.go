package engine

import (
	"github.com/mumax/3/data"
)

func init() {
	DeclFunc("NewSlice", NewSlice, "Makes a 3D array of scalars with given x,y,z size")
}

func NewSlice(Nx, Ny, Nz int) *data.Slice {
	const c = 1 // dummy cell size
	mesh := data.NewMesh(Nz, Ny, Nx, c, c, c)
	return data.NewSlice(1, mesh)
}
