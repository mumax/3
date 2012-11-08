package conv

import (
	"code.google.com/p/nimble-cube/core"
	"code.google.com/p/nimble-cube/gpu"
	"code.google.com/p/nimble-cube/gpu/ptx"
	"github.com/barnex/cuda5/cu"
	"github.com/barnex/cuda5/safe"
	"unsafe"
)

var (
	kernMulRSymm2DxCode  cu.Function
	kernMulRSymm2DyzCode cu.Function
)

func kernMulRSymm2Dyz(fftMy, fftMz safe.Complex64s, K11, K22, K12 safe.Float32s, N1, N2 int, stream cu.Stream) {

	core.Assert(K11.Len() == (N1/2+1)*N2)

	if kernMulRSymm2DyzCode == 0 {
		mod := cu.ModuleLoadData(ptx.KERNMULRSYMM2DYZ)
		kernMulRSymm2DyzCode = mod.GetFunction("kernmulRSymm2Dyz")
	}

	gridDim, blockDim := gpu.Make2DConf(N1, N2)

	m1ptr := fftMy.Pointer()
	m2ptr := fftMz.Pointer()
	k1ptr := K11.Pointer()
	k2ptr := K22.Pointer()
	k3ptr := K12.Pointer()

	args := []unsafe.Pointer{
		unsafe.Pointer(&m1ptr),
		unsafe.Pointer(&m2ptr),
		unsafe.Pointer(&k1ptr),
		unsafe.Pointer(&k2ptr),
		unsafe.Pointer(&k3ptr),
		unsafe.Pointer(&N1),
		unsafe.Pointer(&N2)}

	shmem := 0
	cu.LaunchKernel(kernMulRSymm2DyzCode, gridDim.X, gridDim.Y, gridDim.Z, blockDim.X, blockDim.Y, blockDim.Z, shmem, stream, args)
}

func kernMulRSymm2Dx(fftMx safe.Complex64s, K00 safe.Float32s, N1, N2 int, stream cu.Stream) {

	core.Assert(K00.Len() == (N1/2+1)*N2)

	if kernMulRSymm2DxCode == 0 {
		mod := cu.ModuleLoadData(ptx.KERNMULRSYMM2DX)
		kernMulRSymm2DxCode = mod.GetFunction("kernmulRSymm2Dx")
	}

	gridDim, blockDim := gpu.Make2DConf(N1, N2)

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

//var kernMulRSymm2DCode cu.Function
//
//// Kernel multiplication with purely real, symmetric kernel.
//func kernMulRSymm2D(fftM [3]safe.Complex64s, K00, K11, K22, K12 safe.Float32s, N1, N2 int, stream cu.Stream) {
//
//	core.Assert(K00.Len() == (N1/2+1)*N2)
//
//	if kernMulRSymm2DCode == 0 {
//		mod := cu.ModuleLoadData(ptx.KERNMULRSYMM2D)
//		kernMulRSymm2DCode = mod.GetFunction("kernmulRSymm2D")
//	}
//
//	gridDim, blockDim := gpu.Make2DConf(N1, N2)
//
//	m0ptr := fftM[0].Pointer()
//	m1ptr := fftM[1].Pointer()
//	m2ptr := fftM[2].Pointer()
//	k0ptr := K00.Pointer()
//	k1ptr := K11.Pointer()
//	k2ptr := K22.Pointer()
//	k3ptr := K12.Pointer()
//
//	args := []unsafe.Pointer{
//		unsafe.Pointer(&m0ptr),
//		unsafe.Pointer(&m1ptr),
//		unsafe.Pointer(&m2ptr),
//		unsafe.Pointer(&k0ptr),
//		unsafe.Pointer(&k1ptr),
//		unsafe.Pointer(&k2ptr),
//		unsafe.Pointer(&k3ptr),
//		unsafe.Pointer(&N1),
//		unsafe.Pointer(&N2)}
//
//	shmem := 0
//	cu.LaunchKernel(kernMulRSymm2DCode, gridDim.X, gridDim.Y, gridDim.Z, blockDim.X, blockDim.Y, blockDim.Z, shmem, stream, args)
//}
//
