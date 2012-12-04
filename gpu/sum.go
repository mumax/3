package gpu

import (
	"code.google.com/p/nimble-cube/core"
	"code.google.com/p/nimble-cube/nimble"
	"code.google.com/p/nimble-cube/uni"
	"github.com/barnex/cuda5/cu"
	"github.com/barnex/cuda5/safe"
	"unsafe"
)

func NewSum(tag string, term1, term2 nimble.Chan, weight1, weight2 float32, mem nimble.MemType) *uni.Sum {
	return uni.NewSum(tag, term1, term2, weight1, weight2, mem, GPUDevice)
}

func Madd2Sync(dst, src1, src2 safe.Float32s, factor1, factor2 float32, stream cu.Stream) {
	Madd2Async(dst, src1, src2, factor1, factor2, stream)
	stream.Synchronize()
}

func Madd2Async(dst, src1, src2 safe.Float32s, factor1, factor2 float32, stream cu.Stream) {

	core.Assert(dst.Len() == src1.Len() && dst.Len() == src2.Len())

	maddCode := PTXLoad("madd2")

	N := dst.Len()
	gridDim, blockDim := Make1DConf(N)

	dstptr := dst.Pointer()
	src1ptr := src1.Pointer()
	src2ptr := src2.Pointer()

	args := []unsafe.Pointer{
		unsafe.Pointer(&dstptr),
		unsafe.Pointer(&src1ptr),
		unsafe.Pointer(&factor1),
		unsafe.Pointer(&src2ptr),
		unsafe.Pointer(&factor2),
		unsafe.Pointer(&N)}

	shmem := 0
	cu.LaunchKernel(maddCode, gridDim.X, gridDim.Y, gridDim.Z, blockDim.X, blockDim.Y, blockDim.Z, shmem, stream, args)
}

func Madd3Sync(dst, src1, src2, src3 safe.Float32s, factor1, factor2, factor3 float32, stream cu.Stream) {
	Madd3Async(dst, src1, src2, src3, factor1, factor2, factor3, stream)
	stream.Synchronize()
}

func Madd3Async(dst, src1, src2, src3 safe.Float32s, factor1, factor2, factor3 float32, stream cu.Stream) {

	core.Assert(dst.Len() == src1.Len() && dst.Len() == src2.Len())

	maddCode := PTXLoad("madd3")

	N := dst.Len()
	gridDim, blockDim := Make1DConf(N)

	dstptr := dst.Pointer()
	src1ptr := src1.Pointer()
	src2ptr := src2.Pointer()
	src3ptr := src3.Pointer()

	args := []unsafe.Pointer{
		unsafe.Pointer(&dstptr),
		unsafe.Pointer(&src1ptr),
		unsafe.Pointer(&factor1),
		unsafe.Pointer(&src2ptr),
		unsafe.Pointer(&factor2),
		unsafe.Pointer(&src3ptr),
		unsafe.Pointer(&factor3),
		unsafe.Pointer(&N)}

	shmem := 0
	cu.LaunchKernel(maddCode, gridDim.X, gridDim.Y, gridDim.Z, blockDim.X, blockDim.Y, blockDim.Z, shmem, stream, args)
}
