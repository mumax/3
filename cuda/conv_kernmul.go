package cuda

// Kernel multiplication for purely real kernel, symmetric around Y axis (apart from first row).
// Launch configs range over all complex elements of fft input. This could be optimized: range only over kernel.

import (
	"github.com/mumax/3/data"
	"github.com/mumax/3/util"
)

func kernMulRSymm2Dxy_async(fftMx, fftMy, K11, K22, K12 *data.Slice, Nx, Ny int, str int) {
	util.Argument(K11.Len() == (Ny/2+1)*Nx)
	util.Argument(fftMy.NComp() == 1 && K11.NComp() == 1)

	cfg := make3DConf([3]int{1, Ny, Nx})

	k_kernmulRSymm2Dxy_async(fftMx.DevPtr(0), fftMy.DevPtr(0),
		K11.DevPtr(0), K22.DevPtr(0), K12.DevPtr(0),
		Nx, Ny, cfg, 0)
}

func kernMulRSymm2Dz_async(fftMz, K00 *data.Slice, Nx, Ny int, str int) {
	util.Argument(K00.Len() == (Ny/2+1)*Nx)
	util.Argument(fftMz.NComp() == 1 && K00.NComp() == 1)

	cfg := make3DConf([3]int{1, Ny, Nx})

	k_kernmulRSymm2Dz_async(fftMz.DevPtr(0), K00.DevPtr(0), Nx, Ny, cfg, str)
}

// Does not yet use Y mirror symmetry!!
// Even though it is implemented partially in kernel
func kernMulRSymm3D_async(fftM [3]*data.Slice, K00, K11, K22, K12, K02, K01 *data.Slice, N0, N1, N2 int, str int) {
	util.Argument(K00.Len() == N0*(N1)*N2) // no symmetry yet
	util.Argument(fftM[0].NComp() == 1 && K00.NComp() == 1)

	cfg := make3DConf([3]int{N0, N1, N2})

	k_kernmulRSymm3D_async(fftM[0].DevPtr(0), fftM[1].DevPtr(0), fftM[2].DevPtr(0),
		K00.DevPtr(0), K11.DevPtr(0), K22.DevPtr(0), K12.DevPtr(0), K02.DevPtr(0), K01.DevPtr(0),
		N0, N1, N2, cfg, str)
}
