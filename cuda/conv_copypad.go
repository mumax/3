package cuda

import (
	"code.google.com/p/mx3/data"
	"code.google.com/p/mx3/kernel"
	"code.google.com/p/mx3/util"
)

// Copies src into dst, which is larger or smaller.
// The remainder of dst is not filled with zeros.
func copyPad(dst, src *data.Slice) {
	util.Argument(dst.NComp() == 1 && src.NComp() == 1)
	dstsize := dst.Mesh().Size()
	srcsize := src.Mesh().Size()

	NO := IMin(dstsize[1], srcsize[1])
	N1 := IMin(dstsize[2], srcsize[2])
	gridSize, blockSize := Make2DConf(NO, N1)

	kernel.K_copypad(dst.DevPtr(0), dstsize[0], dstsize[1], dstsize[2],
		src.DevPtr(0), srcsize[0], srcsize[1], srcsize[2],
		gridSize, blockSize)
}
