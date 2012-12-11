package gpu

import (
	"code.google.com/p/nimble-cube/core"
	"github.com/barnex/cuda5/cu"
	"github.com/barnex/cuda5/safe"
	"unsafe"
)

func kernMulRSymm3D(fftM [3]safe.Complex64s, K00, K11, K22, K12, K02, K01 safe.Float32s, N0, N1, N2 int, stream cu.Stream) {
	core.Assert(K11.Len() == N0*N1*N2)

	gridDim, blockDim := Make2DConf(N1, N2)

	m0ptr := fftM[0].Pointer()
	m1ptr := fftM[1].Pointer()
	m2ptr := fftM[2].Pointer()
	k0ptr := K00.Pointer()
	k1ptr := K11.Pointer()
	k2ptr := K22.Pointer()
	k3ptr := K12.Pointer()
	k4ptr := K02.Pointer()
	k5ptr := K01.Pointer()

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
		unsafe.Pointer(&N0),
		unsafe.Pointer(&N1),
		unsafe.Pointer(&N2)}

	shmem := 0
	kernMulRSymm2DyzCode := PTXLoad("kernmulRSymm3D")
	cu.LaunchKernel(kernMulRSymm2DyzCode, gridDim.X, gridDim.Y, gridDim.Z, blockDim.X, blockDim.Y, blockDim.Z, shmem, stream, args)
}
