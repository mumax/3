package cuda

import (
	"github.com/mumax/3/data"
)

// set dst such that its elements are uniform from -Sd/2 to Sd/2 along each axis
func PosScaled(dst *data.Slice, Sx, Sy, Sz float64) {
	N := dst.Len()
	cfg := make1DConf(dst.Len())

	dims := dst.Size()

	Nxf := float32(dims[X])
	Nyf := float32(dims[Y])
	Nzf := float32(dims[Z])

	k_posscaled_async(
		dst.DevPtr(X), dst.DevPtr(Y), dst.DevPtr(Z),
		Nxf, Nyf, Nzf,
		float32(Sx), float32(Sy), float32(Sz),
		N, cfg)
}
