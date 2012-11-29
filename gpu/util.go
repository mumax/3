package gpu

import (
	"code.google.com/p/nimble-cube/core"
	"github.com/barnex/cuda5/cu"
)

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
