package conv

import (
	"github.com/barnex/cuda4/cu"
	"github.com/barnex/cuda4/safe"
	"nimble-cube/core"
	"nimble-cube/gpu"
	"nimble-cube/gpu/ptx"
	"unsafe"
)

var kernMulRSymm2DCode cu.Function

// Kernel multiplication with purely real, symmetric kernel.
func kernMulRSymm2D(fftM [3]safe.Complex64s, K00, K11, K22, K12 safe.Float32s, N1, N2 int, stream cu.Stream) {

	core.Assert(fftM[0].Len() == K00.Len())
	core.Assert(fftM[0].Len() == N1*N2)

	if kernMulRSymm2DCode == 0 {
		mod := cu.ModuleLoadData(ptx.KERNMULRSYMM2D)
		kernMulRSymm2DCode = mod.GetFunction("kernmulRSymm2D")
	}

	gridDim, blockDim := gpu.Make2DConf(N1/2+1, N2)

	m0ptr := fftM[0].Pointer()
	m1ptr := fftM[1].Pointer()
	m2ptr := fftM[2].Pointer()
	k0ptr := K00.Pointer()
	k1ptr := K11.Pointer()
	k2ptr := K22.Pointer()
	k3ptr := K12.Pointer()

	args := []unsafe.Pointer{
		unsafe.Pointer(&m0ptr),
		unsafe.Pointer(&m1ptr),
		unsafe.Pointer(&m2ptr),
		unsafe.Pointer(&k0ptr),
		unsafe.Pointer(&k1ptr),
		unsafe.Pointer(&k2ptr),
		unsafe.Pointer(&k3ptr),
		unsafe.Pointer(&N1),
		unsafe.Pointer(&N2)}

	shmem := 0
	cu.LaunchKernel(kernMulRSymm2DCode, gridDim.X, gridDim.Y, gridDim.Z, blockDim.X, blockDim.Y, blockDim.Z, shmem, stream, args)
}
