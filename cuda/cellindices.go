package cuda

import (
	"github.com/mumax/3/data"
)

// set dst to a 4D slice (comp, ix, iy, iz) containing the <comp>-index of cell (<ix>, <iy>, <iz>)
func CellIndices(dst *data.Slice) {
	N := dst.Len()
	dims := dst.Size()
	nx := float32(dims[X])
	ny := float32(dims[Y])
	nz := float32(dims[Z])

	cfg := make1DConf(dst.Len())
	k_cellindices_async(
		dst.DevPtr(X), dst.DevPtr(Y), dst.DevPtr(Z),
		nx, ny, nz,
		N, cfg)
}
