package gpu

import (
	"github.com/barnex/cuda5/cu"
	"github.com/barnex/cuda5/safe"
	"nimble-cube/core"
	"nimble-cube/gpu/ptx"
	"unsafe"
)

var rotatevecCode cu.Function

// Rotate unit vectors v by factor * delta.
func rotatevec(vec, delta [3]safe.Float32s, factor float32, stream cu.Stream) {
	core.Assert(vec[0].Len() == delta[0].Len())

	if rotatevecCode == 0 {
		mod := cu.ModuleLoadData(ptx.ROTATEVEC)
		rotatevecCode = mod.GetFunction("rotatevec")
	}

	N := vec[0].Len()
	gridDim, blockDim := Make1DConf(N)

	v0ptr := vec[0].Pointer()
	v1ptr := vec[1].Pointer()
	v2ptr := vec[2].Pointer()
	d0ptr := delta[0].Pointer()
	d1ptr := delta[1].Pointer()
	d2ptr := delta[2].Pointer()

	args := []unsafe.Pointer{
		unsafe.Pointer(&v0ptr),
		unsafe.Pointer(&v1ptr),
		unsafe.Pointer(&v2ptr),
		unsafe.Pointer(&d0ptr),
		unsafe.Pointer(&d1ptr),
		unsafe.Pointer(&d2ptr),
		unsafe.Pointer(&factor),
		unsafe.Pointer(&N)}

	shmem := 0
	cu.LaunchKernel(rotatevecCode, gridDim.X, gridDim.Y, gridDim.Z, blockDim.X, blockDim.Y, blockDim.Z, shmem, stream, args)
	stream.Synchronize()
}
