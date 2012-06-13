package nc

// Copies from gpu.

import (
//"github.com/barnex/cuda4/cu"
//"unsafe"
)

type FromGpuBox struct {
	Input  <-chan GpuFloats
	Output []chan<- []float32
}

func NewFromGpuBox() *FromGpuBox {
	box := new(FromGpuBox)
	Register(box)
	return box
}

func (box *FromGpuBox) Run() {
	for {
		in := RecvGpu(box.Input)
		buffer := Buffer()
		//cu.MemcpyDtoH(unsafe.Pointer(&buffer[0]), cu.DevicePtr(in), cu.SIZEOF_FLOAT32*int64(WarpLen()))
		RecycleGpu(in)
		Send(box.Output, buffer)
	}
}
