package nc

// Copies from gpu.

import (
	"github.com/barnex/cuda4/cu"
	"unsafe"
)

type FromGpuBox struct {
	Input  <-chan GpuFloats
	Output []chan<- []float32
	stream cu.Stream
}

func NewFromGpuBox() *FromGpuBox {
	box := new(FromGpuBox)
	box.stream = cu.StreamCreate()
	Register(box)
	return box
}

func (box *FromGpuBox) Run() {
	SetCudaCtx()
	for {
		in := RecvGpu(box.Input)
		buffer := Buffer()
		cu.MemcpyDtoHAsync(unsafe.Pointer(&buffer[0]), cu.DevicePtr(in),
			cu.SIZEOF_FLOAT32*int64(WarpLen()), box.stream)
		box.stream.Synchronize()
		RecycleGpu(in)
		Send(box.Output, buffer)
	}
}
