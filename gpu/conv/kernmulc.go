package conv

import (
	"github.com/barnex/cuda5/cu"
	"github.com/barnex/cuda5/safe"
	"nimble-cube/gpu"
	"nimble-cube/gpu/ptx"
	"nimble-cube/nimble"
	"unsafe"
)

var kernMulCCode cu.Function

// General kernel multiplication with general complex kernel.
// (stored in interleaved format).
// It might be more clear if the kernel were stored as safe.Complex64s.
func kernMulC(fftM [3]safe.Complex64s, K [3][3]safe.Float32s, stream cu.Stream) {

	nimble.Assert(2*fftM[0].Len() == K[0][0].Len())

	if kernMulCCode == 0 {
		mod := cu.ModuleLoadData(ptx.KERNMULC) // TODO: target higher SM's as well.
		kernMulCCode = mod.GetFunction("kernmulC")
	}

	N := fftM[0].Len()
	gridDim, blockDim := gpu.Make1DConf(N)

	m0ptr := fftM[0].Pointer()
	m1ptr := fftM[1].Pointer()
	m2ptr := fftM[2].Pointer()
	k0ptr := K[0][0].Pointer()
	k1ptr := K[1][1].Pointer()
	k2ptr := K[2][2].Pointer()
	k3ptr := K[1][2].Pointer()
	k4ptr := K[0][2].Pointer()
	k5ptr := K[0][1].Pointer()
	k6ptr := K[2][1].Pointer()
	k7ptr := K[2][0].Pointer()
	k8ptr := K[1][0].Pointer()

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
		unsafe.Pointer(&k6ptr),
		unsafe.Pointer(&k7ptr),
		unsafe.Pointer(&k8ptr),
		unsafe.Pointer(&N)}

	shmem := 0
	cu.LaunchKernel(kernMulCCode, gridDim.X, gridDim.Y, gridDim.Z, blockDim.X, blockDim.Y, blockDim.Z, shmem, stream, args)
}
