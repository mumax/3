package nc

// Copies to gpu.

import (
	"github.com/barnex/cuda4/cu"
)

type ToGpuBox struct {
	Input  <-chan Block
	Output []chan<- GpuBlock
	stream cu.Stream
}

func NewToGpuBox() *ToGpuBox {
	box := new(ToGpuBox)
	box.stream = cu.StreamCreate()
	Register(box)
	return box
}

func (box *ToGpuBox) Run() {
	LockCudaCtx()
	for {
		in := Recv(box.Input)
		sendToGpu(in, box.Output, box.stream)
	}
}

func sendToGpu(in Block, out []chan<- GpuBlock, stream cu.Stream) {
	buffer := GpuBuffer()
	buffer.CopyDtoHAsync(in, stream)
	//cu.MemcpyHtoDAsync(buffer.Pointer(), in.UnsafePointer(), in.Bytes(), stream)
	stream.Synchronize()
	Recycle(in)
	SendGpu(out, buffer)
}
