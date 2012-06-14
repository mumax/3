package nc

import (
	"github.com/barnex/cuda4/cu"
	"unsafe"
)

// Copies a 3-vector to GPU,
// component by-component as much possible.
// I.e.: Tries to copy first all X, then Y and finally Z components.
// However, if no X component is ready to be sent, the box will not idle
// but rather start copying the next component.
type ToGpu3SeqBox struct {
	Input  [3]<-chan []float32
	Output [3][]chan<- GpuFloats
}

func NewToGpu3SeqBox() *ToGpu3SeqBox {
	box := new(ToGpu3SeqBox)
	//box.stream = cu.StreamCreate()
	Register(box)
	return box
}

func (box *ToGpu3SeqBox) Run() {
	for {
		in := Recv(box.Input)
		buffer := GpuBuffer()
		SetCudaCtx()
		cu.MemcpyHtoDAsync(cu.DevicePtr(buffer), unsafe.Pointer(&in[0]),
			cu.SIZEOF_FLOAT32*int64(WarpLen()), box.stream)
		box.stream.Synchronize()
		Recycle(in)
		SendGpu(box.Output, buffer)
	}
}
