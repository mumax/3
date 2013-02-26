package cuda

import (
	"code.google.com/p/mx3/data"
	"code.google.com/p/mx3/kernel"
	"code.google.com/p/mx3/util"
	"github.com/barnex/cuda5/cu"
)

// Copies src into dst, which is larger or smaller.
// The remainder of dst is not filled with zeros.
func copyPad(dst, src *data.Slice, dstsize, srcsize [3]int, str cu.Stream) {
	util.Argument(dst.NComp() == 1 && src.NComp() == 1)
	util.Assert(dst.Len() == prod(dstsize))
	util.Assert(src.Len() == prod(srcsize))

	N0 := iMin(dstsize[1], srcsize[1])
	N1 := iMin(dstsize[2], srcsize[2])
	gridSize, blockSize := Make2DConf(N0, N1)

	kernel.K_copypad_async(dst.DevPtr(0), dstsize[0], dstsize[1], dstsize[2],
		src.DevPtr(0), srcsize[0], srcsize[1], srcsize[2],
		gridSize, blockSize, str)
}
