package nc

import (
	"github.com/barnex/cuda4/cu"
)

// Block of float32's on the GPU.
type GpuBlock struct {
	ptr  cu.DevicePtr
	size [3]int
}

func MakeGpuBlock(size [3]int) GpuBlock {
	SetCudaCtx()
	N := size[0] * size[1] * size[2]
	return GpuBlock{cu.MemAlloc(cu.SIZEOF_FLOAT32 * int64(N)), size}
}

func (b *GpuBlock) Pointer() cu.DevicePtr {
	return b.ptr
}

// Total number of scalar elements.
func (b *GpuBlock) NFloat() int {
	return b.size[0] * b.size[1] * b.size[2]
}

// BlockSize is the size of the block (N0, N1, N2)
// as was passed to MakeBlock()
func (b *GpuBlock) BlockSize() [3]int {
	return b.size
}
