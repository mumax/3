package engine

import (
	"github.com/mumax/3/cuda"
	"github.com/mumax/3/data"
)

func init() {
	DeclFunc("CellIndices", CellIndices, "4D slice containing the index of each cell as (ix, iy, iz).")
}

func CellIndices() *data.Slice {
	n := Mesh().Size()
	dst := cuda.NewSlice(3, [3]int{n[X], n[Y], n[Z]})
	cuda.CellIndices(dst)
	return dst.HostCopy()
}
