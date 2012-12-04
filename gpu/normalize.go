package gpu

import (
	"github.com/barnex/cuda5/cu"
	"github.com/barnex/cuda5/safe"
	"unsafe"
)

func NormalizeSync(vec [3]safe.Float32s, stream cu.Stream) {
	N := vec[0].Len()
	gridDim, blockDim := Make1DConf(N)

	v0ptr := vec[0].Pointer()
	v1ptr := vec[1].Pointer()
	v2ptr := vec[2].Pointer()

	args := []unsafe.Pointer{
		unsafe.Pointer(&v0ptr),
		unsafe.Pointer(&v1ptr),
		unsafe.Pointer(&v2ptr),
		unsafe.Pointer(&N)}

	shmem := 0
	code := PTXLoad("normalize")
	cu.LaunchKernel(code, gridDim.X, gridDim.Y, gridDim.Z, blockDim.X, blockDim.Y, blockDim.Z, shmem, stream, args)
	stream.Synchronize()
}
