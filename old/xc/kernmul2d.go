package xc

import (
	"github.com/barnex/cuda4/cu"
	"github.com/barnex/cuda4/safe"
	"nimble-cube/core"
	"nimble-cube/ptx"
	"unsafe"
)

var kernMul2DKern cu.Function

func kernMul2D(fftM [3]safe.Complex64s, K00, K11, K22, K12 safe.Float32s, kernsize [3]int, stream cu.Stream) {

	//core.Assert(fftM[0].Len() >  K00.Len())

	if kernMul2DKern == 0 {
		mod := cu.ModuleLoadData(ptx.KERNMUL2D)
		kernMul2DKern = mod.GetFunction("kernmul2D")
	}

	N1 := kernsize[1]
	N2 := kernsize[2]
	core.Assert(N1*N2 == fftM[0].Len())

	// Launch config: 2D, loop over rows.
	var gridDim, blockDim cu.Dim3
	{
		BLOCKSIZE := 512
		for BLOCKSIZE > N2 {
			BLOCKSIZE /= 2
		}

		gridDim.X = DivUp(N2, BLOCKSIZE)
		gridDim.Y = N1/2 + 1 // bottom half is symmetric, apart from 1st row
		gridDim.Z = 1

		blockDim.X = BLOCKSIZE
		blockDim.Y = 1
		blockDim.Z = 1
	}

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
	cu.LaunchKernel(kernMul2DKern, gridDim.X, gridDim.Y, gridDim.Z, blockDim.X, blockDim.Y, blockDim.Z, shmem, stream, args)
}
