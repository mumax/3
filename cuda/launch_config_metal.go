//go:build darwin && arm64
// +build darwin,arm64

package cuda

// Apple GPUs execute 32-wide SIMD groups. A 32x8 tile keeps X-contiguous
// memory accesses coalesced while using 256 threads, leaving more register
// headroom than CUDA's 1024-thread 32x32 tile. Reduction kernels retain their
// separately defined 512-thread tree.
const (
	BlockSize    = 256
	TileX, TileY = 32, 8
)
