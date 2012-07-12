package xc

import (
	"github.com/barnex/cuda4/cu"
	"github.com/barnex/cuda4/safe"
	"nimble-cube/core"
	"nimble-cube/ptx"
	"unsafe"
)

var kernMulKern cu.Function

func kernMul(fftM [3]safe.Complex64s, K00, K11, K22, K12, K02, K01 safe.Float32s, stream cu.Stream) {

	core.Debug(fftM[0].Len(), K00.Len())
	core.Assert(fftM[0].Len() == K00.Len())

	if kernMulKern == 0 {
		mod := cu.ModuleLoadData(ptx.KERNMUL) // TODO: target higher SM's as well.
		kernMulKern = mod.GetFunction("kernmul")
	}

	N := fftM[0].Len()
	gridDim, blockDim := Make1DConf(N)

	m0ptr := fftM[0].Pointer()
	m1ptr := fftM[1].Pointer()
	m2ptr := fftM[2].Pointer()
	k0ptr := K00.Pointer()
	k1ptr := K11.Pointer()
	k2ptr := K22.Pointer()
	k3ptr := K12.Pointer()
	k4ptr := K02.Pointer()
	k5ptr := K01.Pointer()

	core.Debug(m0ptr, m1ptr, m2ptr, k0ptr, k1ptr, k2ptr, k3ptr, k4ptr, k5ptr)

	args := []unsafe.Pointer{
		unsafe.Pointer(&m0ptr),
		unsafe.Pointer(&m1ptr),
		unsafe.Pointer(&m2ptr),
		unsafe.Pointer(&k0ptr),
		unsafe.Pointer(&k1ptr),
		unsafe.Pointer(&k2ptr),
		unsafe.Pointer(&k3ptr),
		unsafe.Pointer(&k4ptr),
		unsafe.Pointer(&k5ptr),
		unsafe.Pointer(&N)}

	shmem := 0
	core.Debug("cu.LaunchKernel", kernMulKern, gridDim.X, gridDim.Y, gridDim.Z, blockDim.X, blockDim.Y, blockDim.Z, shmem, stream, args)
	cu.LaunchKernel(kernMulKern, gridDim.X, gridDim.Y, gridDim.Z, blockDim.X, blockDim.Y, blockDim.Z, shmem, stream, args)
}
