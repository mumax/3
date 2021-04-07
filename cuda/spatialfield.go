package cuda

import (
	"github.com/mumax/3/data"
)

// set dst such that its elements are evenly spaced by sx, sy, sz along each axis x, y, z and centered around 0, 0, 0
func SpatialField(dst *data.Slice, sx, sy, sz float32) {

	N := dst.Len()
	dims := dst.Size()

	nx := float32(dims[X])
	ny := float32(dims[Y])
	nz := float32(dims[Z])

	cfg := make1DConf(dst.Len())

	k_spatialfield_async(
		dst.DevPtr(X), dst.DevPtr(Y), dst.DevPtr(Z),
		nx, ny, nz,
		sx, sy, sz,
		N, cfg)
}
