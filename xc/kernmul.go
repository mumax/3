package xc

import (
	"github.com/barnex/cuda4/cu"
	//"github.com/barnex/cuda4/safe"
	//"nimble-cube/core"
	"nimble-cube/ptx"
	//"unsafe"
)

var kernMulKern cu.Function

func kernMul() {

	if kernMulKern == 0 {
		mod := cu.ModuleLoadData(ptx.KERNMUL) // TODO: target higher SM's as well.
		kernMulKern = mod.GetFunction("kernmul")
	}

	//	dstptr := dst.Pointer()
	//	srcptr := src.Pointer()
	//
	//	block := 16
	//	gridJ := DivUp(min(dstsize[1], srcsize[1]), block)
	//	gridK := DivUp(min(dstsize[2], srcsize[2]), block)
	//	shmem := 0
	//	args := []unsafe.Pointer{
	//		unsafe.Pointer(&dstptr),
	//		unsafe.Pointer(&dstsize[0]),
	//		unsafe.Pointer(&dstsize[1]),
	//		unsafe.Pointer(&dstsize[2]),
	//		unsafe.Pointer(&srcptr),
	//		unsafe.Pointer(&srcsize[0]),
	//		unsafe.Pointer(&srcsize[1]),
	//		unsafe.Pointer(&srcsize[2]),
	//		unsafe.Pointer(&offset[0]),
	//		unsafe.Pointer(&offset[1]),
	//		unsafe.Pointer(&offset[2])}
	//
	//	cu.LaunchKernel(copyPadKern, gridJ, gridK, 1, block, block, 1, shmem, stream, args)
}
