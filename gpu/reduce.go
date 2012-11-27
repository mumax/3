package gpu

import (
	"github.com/barnex/cuda5/cu"
	"github.com/barnex/cuda5/safe"
	"unsafe"
)

func reduce_sum(in safe.Float32s, stream cu.Stream) float32{
	out := safe.MakeFloat32s(1)
	defer out.Free()

	N := in.Len()

	var gridDim, blockDim cu.Dim3
	blockDim.X = 512
	gridDim.X = 8 // TODO

	inptr := in.Pointer()
	outptr := unsafe.Pointer(uintptr(out.Pointer()))
	args := []unsafe.Pointer{
		unsafe.Pointer(&inptr),
		unsafe.Pointer(&outptr),
		unsafe.Pointer(&N)}

	shmem := 0
	code := PTXLoad("reducesum")
	cu.LaunchKernel(code, gridDim.X, gridDim.Y, gridDim.Z, blockDim.X, blockDim.Y, blockDim.Z, shmem, stream, args)

	var result_ [1]float32
	result := result_[:]
	out.CopyDtoHAsync(result, stream)
	stream.Synchronize()
	return result_[0]	
}
