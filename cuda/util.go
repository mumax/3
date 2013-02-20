package cuda

import (
	"code.google.com/p/mx3/util"
	"github.com/barnex/cuda5/cu"
)

//// Undo LockCudaThread()
//func UnlockCudaThread() {
//	if cudaCtx == 0 {
//		return // allow to run if there's no GPU.
//	}
//	runtime.UnlockOSThread()
//	c := atomic.AddInt32(&lockCount, -1)
//	core.Debug("Unlocked OS thread,", c, "remain locked")
//}
//
//// Register host memory for fast transfers,
//// but only when flag -pagelock is true.
//func MemHostRegister(slice []float32) {
//	// do not fail on already registered memory.
//	defer func() {
//		err := recover()
//		if err != nil && err != cu.ERROR_HOST_MEMORY_ALREADY_REGISTERED {
//			panic(err)
//		}
//	}()
//	if *nimble.Flag_pagelock {
//		cu.MemHostRegister(unsafe.Pointer(&slice[0]), cu.SIZEOF_FLOAT32*int64(len(slice)), cu.MEMHOSTREGISTER_PORTABLE)
//	}
//}

func iMin(a, b int) int {
	if a < b {
		return a
	}
	return b
}

// Integer division rounded up.
func divUp(x, y int) int {
	return ((x - 1) / y) + 1
}

// CUDA Launch parameters. TODO: use device properties.
const (
	MaxBlockSize = 512
	MaxGridSize  = 65535
)

// Make a 1D kernel launch configuration suited for N threads.
func Make1DConf(N int) (gridSize, blockSize cu.Dim3) {

	blockSize.X = MaxBlockSize
	blockSize.Y = 1
	blockSize.Z = 1

	N2 := divUp(N, MaxBlockSize) // N2 blocks left

	NX := divUp(N2, MaxGridSize)
	NY := divUp(N2, NX)

	gridSize.X = NX
	gridSize.Y = NY
	gridSize.Z = 1

	util.Assert(gridSize.X*gridSize.Y*gridSize.Z*blockSize.X*blockSize.Y*blockSize.Z >= N)
	return
}

func Make2DConf(N1, N2 int) (gridSize, blockSize cu.Dim3) {
	const BLOCK = 16 // TODO

	blockSize.X = BLOCK
	blockSize.Y = BLOCK
	blockSize.Z = 1

	NX := divUp(N2, BLOCK)
	NY := divUp(N1, BLOCK)

	gridSize.X = NX
	gridSize.Y = NY
	gridSize.Z = 1

	N := N1 * N2
	util.Assert(gridSize.X*gridSize.Y*gridSize.Z*blockSize.X*blockSize.Y*blockSize.Z >= N)
	return
}
