package gpu

import (
	"code.google.com/p/mx3/core"
	"code.google.com/p/mx3/nimble"
	"github.com/barnex/cuda5/cu"
)

var lockCount int32

// To be called by any fresh goroutine that will do cuda interaction.
func LockCudaThread() {
	if cudaCtx == 0 {
		return // allow to run if there's no GPU.
	}
	runtime.LockOSThread()
	cudaCtx.SetCurrent() // super cheap.
	c := atomic.AddInt32(&lockCount, 1)
	core.Debug("Locked thread", c, "to CUDA context")
}

// Undo LockCudaThread()
func UnlockCudaThread() {
	if cudaCtx == 0 {
		return // allow to run if there's no GPU.
	}
	runtime.UnlockOSThread()
	c := atomic.AddInt32(&lockCount, -1)
	core.Debug("Unlocked OS thread,", c, "remain locked")
}

// Register host memory for fast transfers,
// but only when flag -pagelock is true.
func MemHostRegister(slice []float32) {
	// do not fail on already registered memory.
	defer func() {
		err := recover()
		if err != nil && err != cu.ERROR_HOST_MEMORY_ALREADY_REGISTERED {
			panic(err)
		}
	}()
	if *nimble.Flag_pagelock {
		cu.MemHostRegister(unsafe.Pointer(&slice[0]), cu.SIZEOF_FLOAT32*int64(len(slice)), cu.MEMHOSTREGISTER_PORTABLE)
	}
}

func IMin(a, b int) int {
	if a < b {
		return a
	}
	return b
}

// Integer division rounded up.
func DivUp(x, y int) int {
	return ((x - 1) / y) + 1
}

const (
	maxBlockSize = 512   // TODO
	maxGridSize  = 65535 // TODO
)

// Make a 1D kernel launch configuration suited for N threads.
func Make1DConf(N int) (gridSize, blockSize cu.Dim3) {

	blockSize.X = maxBlockSize
	blockSize.Y = 1
	blockSize.Z = 1

	N2 := DivUp(N, maxBlockSize) // N2 blocks left

	NX := DivUp(N2, maxGridSize)
	NY := DivUp(N2, NX)

	gridSize.X = NX
	gridSize.Y = NY
	gridSize.Z = 1

	core.Assert(gridSize.X*gridSize.Y*gridSize.Z*blockSize.X*blockSize.Y*blockSize.Z >= N)
	return
}

func Make2DConf(N1, N2 int) (gridSize, blockSize cu.Dim3) {
	const BLOCK = 16 // TODO

	blockSize.X = BLOCK
	blockSize.Y = BLOCK
	blockSize.Z = 1

	NX := DivUp(N2, BLOCK)
	NY := DivUp(N1, BLOCK)

	gridSize.X = NX
	gridSize.Y = NY
	gridSize.Z = 1

	N := N1 * N2
	core.Assert(gridSize.X*gridSize.Y*gridSize.Z*blockSize.X*blockSize.Y*blockSize.Z >= N)
	return
}

func Copy(dst, src Slice) {
	core.Assert(dst.Len() == src.Len() && dst.NComp() == src.NComp())
	bytes := dst.Bytes()
	str := Stream()
	for c := 0; c < dst.NComp(); c++ {
		cu.MemcpyAsync(dst.DevPtr(c), src.DevPtr(c), bytes, str)
	}
	SyncAndRecycle(str)
}

// Set the entire slice to this value.
func Memset(dst nimble.Slice, val ...float32) {
	core.Assert(len(val) == dst.NComp())
	str := Stream()
	for c, v := range val {
		cu.MemsetD32Async(dst.DevPtr(), math.Float32bits(value), int64(s.Len()), str)
	}
	SyncAndRecycle(str)
}
