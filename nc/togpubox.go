package nc

// Copies to gpu.

import (
	"github.com/barnex/cuda4/cu"
)

type ToGpuBox struct {
	Input  <-chan Block
	Output []chan<- GpuBlock
}

func NewToGpuBox() *ToGpuBox {
	box := new(ToGpuBox)
	Register(box)
	return box
}

func (box *ToGpuBox) Run() {
	LockCudaThread()
	str := cu.StreamCreate()
	defer str.Destroy()
	for {
		in := Recv(box.Input)
		sendToGpu(in, box.Output, str)
	}
}

func sendToGpu(in Block, out []chan<- GpuBlock, stream cu.Stream) {
	buffer := GpuBuffer()
	buffer.CopyHtoDAsync(in, stream)
	stream.Synchronize()
	Recycle(in)
	SendGpu(out, buffer)
}
