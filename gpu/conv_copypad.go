package gpu

import (
	"code.google.com/p/mx3/core"
	"code.google.com/p/mx3/gpu/ptx"
	"github.com/barnex/cuda5/cu"
	"github.com/barnex/cuda5/safe"
)

// Copies src into dst, which is larger or smaller.
// The remainder of dst is not filled with zeros.
func copyPad(dst, src safe.Float32s, dstsize, srcsize, offset [3]int) {
	core.Assert(dst.Len() == core.Prod(dstsize))
	core.Assert(src.Len() == core.Prod(srcsize))

	const block = 16
	gridJ := DivUp(IMin(dstsize[1], srcsize[1]), block)
	gridK := DivUp(IMin(dstsize[2], srcsize[2]), block)
	gridSize := cu.Dim3{gridK, gridJ, 1}
	blockSize := cu.Dim3{block, block, 1}

	ptx.K_copypad(dst.Pointer(), dstsize[0], dstsize[1], dstsize[2],
		src.Pointer(), srcsize[0], srcsize[1], srcsize[2],
		gridSize, blockSize)
}
