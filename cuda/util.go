package cuda

import "github.com/barnex/cuda5/cu"

// CUDA Launch parameters. TODO: optimize?
var (
	BlockSize    = 512
	TileX, TileY = 32, 32
	MaxGridSize  = 65535
)

// cuda launch configuration
type config struct {
	Grid, Block cu.Dim3
}

// Make a 1D kernel launch configuration suited for N threads.
func make1DConf(N int) *config {
	bl := cu.Dim3{X: BlockSize, Y: 1, Z: 1}

	N2 := divUp(N, BlockSize) // N2 blocks left
	NX := divUp(N2, MaxGridSize)
	NY := divUp(N2, NX)
	gr := cu.Dim3{X: NX, Y: NY, Z: 1}

	return &config{gr, bl}
}

// Make a 3D kernel launch configuration suited for N threads.
func make3DConf(N [3]int) *config {
	bl := cu.Dim3{X: TileX, Y: TileY, Z: 1}

	NZ := N[Z]
	NY := divUp(N[Y], TileY)
	NX := divUp(N[X], TileX)
	gr := cu.Dim3{X: NX, Y: NY, Z: NZ}

	dbg("3dconf", N, "=", config{gr, bl})
	return &config{gr, bl}
}

// integer minimum
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

// vector components
const (
	X = 0
	Y = 1
	Z = 2
)
