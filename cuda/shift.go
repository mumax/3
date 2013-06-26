package cuda

import (
	"code.google.com/p/mx3/data"
	"code.google.com/p/mx3/util"
)

// Copy dst to src, shifting data by given number of cells.
// Off-boundary values are clamped. Used, e.g., to make the
// simulation window follow interesting features.
func Shift(dst, src *data.Slice, shift [3]int) {
	util.Argument(dst.NComp() == 1 && src.NComp() == 1)
	util.Assert(dst.Len() == src.Len())

	N := dst.Mesh().Size()
	cfg := make3DConf(N)

	k_shift(dst.DevPtr(0), src.DevPtr(0), N[0], N[1], N[2], shift[0], shift[1], shift[2], cfg)
}
