package nc

import (
	"github.com/barnex/cuda4/cu"
)

// Slice of float32's on the GPU.
type GpuFloats cu.DevicePtr

func MakeGpuFloats(length int) GpuFloats {
	SetCudaCtx()
	return GpuFloats(cu.MemAlloc(cu.SIZEOF_FLOAT32 * int64(length)))
}
