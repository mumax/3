package gpu

import (
	"code.google.com/p/mx3/core"
	"code.google.com/p/mx3/gpu/ptx"
	"github.com/barnex/cuda5/cu"
	"github.com/barnex/cuda5/safe"
	"unsafe"
)

func kernMulRSymm2Dyz(fftMy, fftMz safe.Complex64s, K11, K22, K12 safe.Float32s, N1, N2 int, stream cu.Stream) {
	core.Assert(K11.Len() == (N1/2+1)*N2)
	gridDim, blockDim := Make2DConf(N1, N2)
	ptx.K_kernmulRSymm2Dyz(fftMy.Pointer(), fftMz.Pointer(), K11.Pointer(), K22.Pointer(), K12.Pointer(), N1, N2, gridDim, blockDim)
}

func kernMulRSymm2Dx(fftMx safe.Complex64s, K00 safe.Float32s, N1, N2 int, stream cu.Stream) {

	core.Assert(K00.Len() == (N1/2+1)*N2)

	kernMulRSymm2DxCode := PTXLoad("kernmulRSymm2Dx")

	gridDim, blockDim := Make2DConf(N1, N2)

	m0ptr := fftMx.Pointer()
	k0ptr := K00.Pointer()

	args := []unsafe.Pointer{
		unsafe.Pointer(&m0ptr),
		unsafe.Pointer(&k0ptr),
		unsafe.Pointer(&N1),
		unsafe.Pointer(&N2)}

	shmem := 0
	cu.LaunchKernel(kernMulRSymm2DxCode, gridDim.X, gridDim.Y, gridDim.Z, blockDim.X, blockDim.Y, blockDim.Z, shmem, stream, args)
}
