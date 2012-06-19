package nc

import (
	"github.com/barnex/cuda4/cu"
)

// Block of float32's on the GPU.
type GpuBlock struct {
	Ptr  cu.DevicePtr
	Size [3]int
}

func MakeGpuBlock(size [3]int) GpuBlock {
	SetCudaCtx()
	N := size[0] * size[1] * size[2]
	return GpuBlock{Ptr: cu.MemAlloc(cu.SIZEOF_FLOAT32 * int64(N)), Size: size}
}
