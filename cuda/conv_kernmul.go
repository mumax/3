package cuda

// Kernel multiplication for purely real kernel, symmetric around Y axis (apart from first row).
// Launch configs range over all complex elements of fft input. This could be optimized: range only over kernel.

import (
	"github.com/mumax/3/data"
	"github.com/mumax/3/util"
)

func kernMulRSymm2Dxy_async(fftMx, fftMy, Kxx, Kyy, Kxy *data.Slice, Nx, Ny int, str int) {
	util.Argument(Kxx.Len() == (Ny/2+1)*Nx)
	util.Argument(fftMy.NComp() == 1 && Kxx.NComp() == 1)

	cfg := make3DConf([3]int{Nx, Ny, 1})

	k_kernmulRSymm2Dxy_async(fftMx.DevPtr(0), fftMy.DevPtr(0),
		Kxx.DevPtr(0), Kxx.DevPtr(0), Kxy.DevPtr(0),
		Nx, Ny, cfg, 0)
}

func kernMulRSymm2Dz_async(fftMz, Kzz *data.Slice, Nx, Ny int, str int) {
	util.Argument(Kzz.Len() == (Ny/2+1)*Nx)
	util.Argument(fftMz.NComp() == 1 && Kzz.NComp() == 1)

	cfg := make3DConf([3]int{Nx, Ny, 1})

	k_kernmulRSymm2Dz_async(fftMz.DevPtr(0), Kzz.DevPtr(0), Nx, Ny, cfg, str)
}

// Does not yet use Y mirror symmetry!!
// Even though it is implemented partially in kernel
func kernMulRSymm3D_async(fftM [3]*data.Slice, Kxx, Kyy, Kzz, Kyz, Kxz, Kxy *data.Slice, Nx, Ny, Nz int, str int) {
	util.Argument(Kxx.Len() == Nx*(Ny)*Nz) // no symmetry yet
	util.Argument(fftM[X].NComp() == 1 && Kxx.NComp() == 1)

	cfg := make3DConf([3]int{Nx, Ny, Nz})

	k_kernmulRSymm3D_async(fftM[X].DevPtr(0), fftM[Y].DevPtr(0), fftM[Z].DevPtr(0),
		Kxx.DevPtr(0), Kyy.DevPtr(0), Kzz.DevPtr(0), Kyz.DevPtr(0), Kxz.DevPtr(0), Kxy.DevPtr(0),
		Nx, Ny, Nz, cfg, str)
}
