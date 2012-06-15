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
		sendToGpu(in, box.Output, box.stream)
	}
}

func sendToGpu(in []float32, out []chan<- GpuFloats, stream cu.Stream) {
	buffer := GpuBuffer()
	SetCudaCtx()
	cu.MemcpyHtoDAsync(cu.DevicePtr(buffer), unsafe.Pointer(&in[0]),
		cu.SIZEOF_FLOAT32*int64(WarpLen()), stream)
	stream.Synchronize()
	Recycle(in)
	SendGpu(out, buffer)
}
