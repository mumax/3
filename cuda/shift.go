package cuda

import (
	"code.google.com/p/mx3/data"
	"code.google.com/p/mx3/util"
)

func Shift(dst, src *data.Slice, shift [3]int) {
	util.Argument(dst.NComp() == 1 && src.NComp() == 1)
	util.Assert(dst.Len() == src.Len())

	N := dst.Mesh().Size()
	cfg := make2DConf(N[2], N[1])

	k_shift(dst.DevPtr(0), src.DevPtr(0), N[0], N[1], N[2], shift[0], shift[1], shift[2], cfg)
}
