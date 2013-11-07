package cuda

import (
	"github.com/mumax/3/data"
	"github.com/mumax/3/util"
)

// Copy dst to src, shifting data by given number of cells.
// Off-boundary values are clamped. Used, e.g., to make the
// simulation window follow interesting features.
func Shift(dst, src *data.Slice, shift [3]int) {
	util.Argument(dst.NComp() == 1 && src.NComp() == 1)
	util.Assert(dst.Len() == src.Len())

	N := dst.Mesh().Size()
	cfg := make3DConf(N)

	k_shift_async(dst.DevPtr(0), src.DevPtr(0), N[X], N[Y], N[Z], shift[X], shift[Y], shift[Z], cfg, stream0)
}

// Like Shift, but for bytes
func ShiftBytes(dst, src *Bytes, m *data.Mesh, shift [3]int) {
	N := m.Size()
	cfg := make3DConf(N)

	k_shiftbytes_async(dst.Ptr, src.Ptr, N[X], N[Y], N[Z], shift[X], shift[Y], shift[Z], cfg, stream0)
}
