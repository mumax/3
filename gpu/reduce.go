package gpu

import (
	"github.com/barnex/cuda5/cu"
	"github.com/barnex/cuda5/safe"
	"unsafe"
)


func reduceSum(in safe.Float32s, stream cu.Stream) float32 {
	return reduce("reducesum", in, stream)
}

func reduceMax(in safe.Float32s, stream cu.Stream) float32 {
	return reduce("reducemax", in, stream)
}

func reduce(op string, in safe.Float32s, stream cu.Stream) float32 {
	out := safe.MakeFloat32s(1)
	defer out.Free()

	N := in.Len()

	blockDim := cu.Dim3{512, 1, 1}
	gridDim := cu.Dim3{8, 1, 1} // 8 is typ. number of multiprocessors.

	inptr := in.Pointer()
	outptr := out.Pointer()
	args := []unsafe.Pointer{
		unsafe.Pointer(&inptr),
		unsafe.Pointer(&outptr),
		unsafe.Pointer(&N)}

	shmem := 0
	code := PTXLoad(op)
	cu.LaunchKernel(code, gridDim.X, gridDim.Y, gridDim.Z, blockDim.X, blockDim.Y, blockDim.Z, shmem, stream, args)
	stream.Synchronize()

	var result_ [1]float32
	result := result_[:]
	out.CopyDtoH(result) // async? register one arena block // , stream)
	return result_[0]
}
