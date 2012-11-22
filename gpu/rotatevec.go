package gpu

import (
	"code.google.com/p/nimble-cube/core"
	"code.google.com/p/nimble-cube/gpu/ptx"
	"github.com/barnex/cuda5/cu"
	"github.com/barnex/cuda5/safe"
	"unsafe"
)

// Rotate unit vectors v by factor * delta.
func rotatevec(vec, delta [3]safe.Float32s, factor float32, stream cu.Stream) {
	core.Assert(vec[0].Len() == delta[0].Len())

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
	rotatevecCode := PTXLoad("rotatevec")
	cu.LaunchKernel(rotatevecCode, gridDim.X, gridDim.Y, gridDim.Z, blockDim.X, blockDim.Y, blockDim.Z, shmem, stream, args)
	stream.Synchronize()
}

var rotatevec2Code cu.Function

// Rotate unit vectors v by factor * delta.
func rotatevec2(vec, delta1 [3]safe.Float32s, factor1 float32, delta2 [3]safe.Float32s, factor2 float32, stream cu.Stream) {
	core.Assert(vec[0].Len() == delta1[0].Len() && vec[0].Len() == delta2[0].Len())

	if rotatevec2Code == 0 {
		mod := cu.ModuleLoadData(ptx.ROTATEVEC2)
		rotatevec2Code = mod.GetFunction("rotatevec2")
	}

	N := vec[0].Len()
	gridDim, blockDim := Make1DConf(N)

	v0ptr := vec[0].Pointer()
	v1ptr := vec[1].Pointer()
	v2ptr := vec[2].Pointer()
	d0ptr := delta1[0].Pointer()
	d1ptr := delta1[1].Pointer()
	d2ptr := delta1[2].Pointer()
	e0ptr := delta2[0].Pointer()
	e1ptr := delta2[1].Pointer()
	e2ptr := delta2[2].Pointer()

	args := []unsafe.Pointer{
		unsafe.Pointer(&v0ptr),
		unsafe.Pointer(&v1ptr),
		unsafe.Pointer(&v2ptr),
		unsafe.Pointer(&d0ptr),
		unsafe.Pointer(&d1ptr),
		unsafe.Pointer(&d2ptr),
		unsafe.Pointer(&factor1),
		unsafe.Pointer(&e0ptr),
		unsafe.Pointer(&e1ptr),
		unsafe.Pointer(&e2ptr),
		unsafe.Pointer(&factor2),
		unsafe.Pointer(&N)}

	shmem := 0
	rotatevec2Code := PTXLoad("rotatevec2")
	cu.LaunchKernel(rotatevec2Code, gridDim.X, gridDim.Y, gridDim.Z, blockDim.X, blockDim.Y, blockDim.Z, shmem, stream, args)
	stream.Synchronize()
}
