package nc

// Copies to gpu.

import (
	"github.com/barnex/cuda4/cu"
	"unsafe"
)

type ToGpuBox struct {
	Input  <-chan []float32
	Output []chan<- GpuFloats
	stream cu.Stream
}

func NewToGpuBox() *ToGpuBox {
	box := new(ToGpuBox)
	box.stream = cu.StreamCreate()
	Register(box)
	return box
}

func (box *ToGpuBox) Run() {
	for {
		in := Recv(box.Input)
		buffer := GpuBuffer()
		cu.MemcpyHtoDAsync(cu.DevicePtr(buffer), unsafe.Pointer(&in[0]),
			cu.SIZEOF_FLOAT32*int64(WarpLen()), box.stream)
		box.stream.Synchronize()
		Recycle(in)
		SendGpu(box.Output, buffer)
	}
}
