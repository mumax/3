package cuda

// Kernel multiplication for purely real kernel, symmetric around Y axis (apart from first row).
// Launch configs range over all complex elements of fft input. This could be optimized: range only over kernel.

import (
	"code.google.com/p/mx3/data"
	"code.google.com/p/mx3/util"
	"github.com/barnex/cuda5/cu"
)

func kernMulRSymm2Dyz(fftMy, fftMz, K11, K22, K12 *data.Slice, N1, N2 int, str cu.Stream) {
	util.Argument(K11.Len() == (N1/2+1)*N2)
	util.Argument(fftMy.NComp() == 1 && K11.NComp() == 1)

	cfg := make2DConf(N1, N2)

	k_kernmulRSymm2Dyz_async(fftMy.DevPtr(0), fftMz.DevPtr(0),
		K11.DevPtr(0), K22.DevPtr(0), K12.DevPtr(0),
		N1, N2, cfg, str)
}

func kernMulRSymm2Dx(fftMx, K00 *data.Slice, N1, N2 int, str cu.Stream) {
	util.Argument(K00.Len() == (N1/2+1)*N2)
	util.Argument(fftMx.NComp() == 1 && K00.NComp() == 1)

	cfg := make2DConf(N1, N2)

	k_kernmulRSymm2Dx_async(fftMx.DevPtr(0), K00.DevPtr(0), N1, N2, cfg, str)
}

// Does not yet use Y mirror symmetry!!
// Even though it is implemented partially in kernel
func kernMulRSymm3D(fftM [3]*data.Slice, K00, K11, K22, K12, K02, K01 *data.Slice, N0, N1, N2 int, str cu.Stream) {
	util.Argument(K00.Len() == N0*(N1)*N2) // no symmetry yet
	util.Argument(fftM[0].NComp() == 1 && K00.NComp() == 1)

	cfg := make3DConf([3]int{N0, N1, N2})

	k_kernmulRSymm3D_async(fftM[0].DevPtr(0), fftM[1].DevPtr(0), fftM[2].DevPtr(0),
		K00.DevPtr(0), K11.DevPtr(0), K22.DevPtr(0), K12.DevPtr(0), K02.DevPtr(0), K01.DevPtr(0),
		N0, N1, N2, cfg, str)
}
