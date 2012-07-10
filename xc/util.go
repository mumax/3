package xc

import (
	"github.com/barnex/cuda4/cu"
	"nimble-cube/core"
)

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

// Integer division rounded up.
func DivUp(x, y int) int {
	return ((x - 1) / y) + 1
}

func Make1DConf(N int) (gridSize, blockSize cu.Dim3) {

	const maxBlockSize = 512
	const maxGridSize = 65535

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
