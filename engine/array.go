package engine

import (
	"github.com/mumax/3/data"
)

func init() {
	DeclFunc("NewArray", NewArray, "Makes a new in-memory array with given number of components and size")
}

func NewArray(nComp, Nx, Ny, Nz int) *data.Slice {
	const c = 1 // dummy cell size
	mesh := data.NewMesh(Nz, Ny, Nx, c, c, c)
	return data.NewSlice(nComp, mesh)
}
