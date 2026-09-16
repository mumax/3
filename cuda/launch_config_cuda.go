//go:build !darwin || !arm64
// +build !darwin !arm64

package cuda

// Preserve the launch geometry used by the upstream CUDA backend.
const (
	BlockSize    = 512
	TileX, TileY = 32, 32
)
