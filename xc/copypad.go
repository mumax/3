package xc

import (
	"github.com/barnex/cuda4/cu"
	"github.com/barnex/cuda4/safe"
	"nimble-cube/core"
	"nimble-cube/ptx"
	"unsafe"
)

var copyPadKern cu.Function

// Copies src into dst (which is larger or smaller), at offset position.
func copyPad(dst, src safe.Float32s, dstsize, srcsize, offset [3]int, stream cu.Stream) {
	core.Assert(dst.Len() == prod(dstsize))
	core.Assert(src.Len() == prod(srcsize))
	// TODO: either remove offset or check offset

	if copyPadKern == 0 {
		mod := cu.ModuleLoadData(ptx.COPYPAD) // TODO: target higher SM's as well.
		copyPadKern = mod.GetFunction("copypad")
	}

	dstptr := dst.Pointer()
	srcptr := src.Pointer()

	block := 16
	gridJ := DivUp(min(dstsize[1], srcsize[1]), block)
	gridK := DivUp(min(dstsize[2], srcsize[2]), block)
	shmem := 0
	args := []unsafe.Pointer{
		unsafe.Pointer(&dstptr),
		unsafe.Pointer(&dstsize[0]),
		unsafe.Pointer(&dstsize[1]),
		unsafe.Pointer(&dstsize[2]),
		unsafe.Pointer(&srcptr),
		unsafe.Pointer(&srcsize[0]),
		unsafe.Pointer(&srcsize[1]),
		unsafe.Pointer(&srcsize[2]),
		unsafe.Pointer(&offset[0]),
		unsafe.Pointer(&offset[1]),
		unsafe.Pointer(&offset[2])}

	cu.LaunchKernel(copyPadKern, gridJ, gridK, 1, block, block, 1, shmem, stream, args)
}
