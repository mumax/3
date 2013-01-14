package gpu

import (
	"code.google.com/p/mx3/core"
	"code.google.com/p/mx3/gpu/ptx"
	"github.com/barnex/cuda5/safe"
)

func kernMulRSymm3D(fftM [3]safe.Complex64s, K00, K11, K22, K12, K02, K01 safe.Float32s, N0, N1, N2 int) {
	core.Assert(K11.Len() == N0*N1*N2)
	gridDim, blockDim := Make2DConf(N1, N2)
	ptx.K_kernmulRSymm3D(fftM[0].Pointer(), fftM[1].Pointer(), fftM[2].Pointer(),
		K00.Pointer(), K11.Pointer(), K22.Pointer(), K12.Pointer(), K02.Pointer(), K01.Pointer(),
		N0, N1, N2, gridDim, blockDim)
}
