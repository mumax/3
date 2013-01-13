package gpu

import (
	"code.google.com/p/mx3/core"
	"code.google.com/p/mx3/gpu/ptx"
	"github.com/barnex/cuda5/safe"
)

func kernMulRSymm2Dyz(fftMy, fftMz safe.Complex64s, K11, K22, K12 safe.Float32s, N1, N2 int) {
	core.Assert(K11.Len() == (N1/2+1)*N2)
	gridDim, blockDim := Make2DConf(N1, N2)
	ptx.K_kernmulRSymm2Dyz(fftMy.Pointer(), fftMz.Pointer(),
		K11.Pointer(), K22.Pointer(), K12.Pointer(),
		N1, N2, gridDim, blockDim)
}

func kernMulRSymm2Dx(fftMx safe.Complex64s, K00 safe.Float32s, N1, N2 int) {
	core.Assert(K00.Len() == (N1/2+1)*N2)
	gridDim, blockDim := Make2DConf(N1, N2)
	ptx.K_kernmulRSymm2Dx(fftMx.Pointer(), K00.Pointer(), N1, N2, gridDim, blockDim)
}
