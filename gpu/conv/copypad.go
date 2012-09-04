package conv

import (
	"github.com/barnex/cuda4/cu"
	"github.com/barnex/cuda4/safe"
	"nimble-cube/core"
	"nimble-cube/gpu"
	"nimble-cube/gpu/ptx"
	"unsafe"
)

var copyPadKern cu.Function

// Copies src into dst (which is larger or smaller), at offset position.
func copyPad(dst, src safe.Float32s, dstsize, srcsize, offset [3]int, stream cu.Stream) {
	core.Debug("copypad", "dstsize", dstsize, "srcsize", srcsize)
	core.Assert(dst.Len() == core.Prod(dstsize))
	core.Assert(src.Len() == core.Prod(srcsize))
	// TODO: either remove offset or check offset

	if copyPadKern == 0 {
		mod := cu.ModuleLoadData(ptx.COPYPAD) // TODO: target higher SM's as well.
		copyPadKern = mod.GetFunction("copypad")
	}

	dstptr := dst.Pointer()
	srcptr := src.Pointer()

	block := 1
	gridJ := gpu.DivUp(gpu.Min(dstsize[1], srcsize[1]), block)
	gridK := gpu.DivUp(gpu.Min(dstsize[2], srcsize[2]), block)
	shmem := 0
	//copypad(float* dst, int D0, int D1, int D2, 
	//        float* src, int S0, int S1, int S2, 
	//        int o0, int o1, int o2){
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

	core.Debug("cu.LaunchKernel", copyPadKern, gridJ, gridK, 1, block, block, 1, shmem, stream, args)
	cu.LaunchKernel(copyPadKern, gridJ, gridK, 1, block, block, 1, shmem, stream, args)
}
