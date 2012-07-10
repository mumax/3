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

	core.Assert(fftM[0].Len() == 2*K00.Len())

	if kernMulKern == 0 {
		mod := cu.ModuleLoadData(ptx.KERNMUL) // TODO: target higher SM's as well.
		kernMulKern = mod.GetFunction("kernmul")
	}

	N := fftM[0].Len()
	gridDim, blockDim := Make1DConf(N)

	args := []unsafe.Pointer{
		unsafe.Pointer(uintptr(fftM[0].Pointer())),
		unsafe.Pointer(uintptr(fftM[1].Pointer())),
		unsafe.Pointer(uintptr(fftM[2].Pointer())),
		unsafe.Pointer(uintptr(K00.Pointer())),
		unsafe.Pointer(uintptr(K11.Pointer())),
		unsafe.Pointer(uintptr(K22.Pointer())),
		unsafe.Pointer(uintptr(K12.Pointer())),
		unsafe.Pointer(uintptr(K02.Pointer())),
		unsafe.Pointer(uintptr(K01.Pointer())),
		unsafe.Pointer(&N)}

	shmem := 0
	cu.LaunchKernel(copyPadKern, gridDim.X, gridDim.Y, gridDim.Z, blockDim.X, blockDim.Y, blockDim.Z, shmem, stream, args)
}
