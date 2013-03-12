package cuda

import "github.com/barnex/cuda5/cu"

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
func make1DConf(N int) *config {

	var gr, bl cu.Dim3
	bl.X = MaxBlockSize
	bl.Y = 1
	bl.Z = 1

	N2 := divUp(N, MaxBlockSize) // N2 blocks left

	NX := divUp(N2, MaxGridSize)
	NY := divUp(N2, NX)

	gr.X = NX
	gr.Y = NY
	gr.Z = 1
	//util.Assert(gridSize.X*gridSize.Y*gridSize.Z*blockSize.X*blockSize.Y*blockSize.Z >= N)

	return &config{gr, bl}
}

// TODO: swap N1/N2?
func make2DConf(N1, N2 int) *config {
	const BLOCK = 32 // TODO

	var gr, bl cu.Dim3
	bl.X = BLOCK
	bl.Y = BLOCK
	bl.Z = 1

	NX := divUp(N2, BLOCK)
	NY := divUp(N1, BLOCK)

	gr.X = NX
	gr.Y = NY
	gr.Z = 1

	//N := N1 * N2
	//util.Assert(gr.X*gr.Y*gr.Z*bl.X*bl.Y*bl.Z >= N)

	return &config{gr, bl}
}

type config struct {
	Grid, Block cu.Dim3
}

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
