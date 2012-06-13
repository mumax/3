package nc

// Copies to gpu.

import (
	"github.com/barnex/cuda4/cu"
	"unsafe"
)

type ToGpuBox struct {
	Input  <-chan []float32
	Output []chan<- GpuFloats
}

func NewToGpuBox() *ToGpuBox {
	box := new(ToGpuBox)
	Register(box)
	return box
}

func (box *ToGpuBox) Run() {
	for {
		in := Recv(box.Input)
		buffer := GpuBuffer()
		cu.MemcpyHtoD(cu.DevicePtr(buffer), unsafe.Pointer(&in[0]), cu.SIZEOF_FLOAT32*int64(WarpLen()))
		SendGpu(box.Output, buffer)
	}
}
