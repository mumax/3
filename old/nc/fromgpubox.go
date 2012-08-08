package nc

// Copies from gpu.

import (
	"github.com/barnex/cuda4/cu"
)

type FromGpuBox struct {
	Input  <-chan GpuBlock
	Output []chan<- Block
}

func NewFromGpuBox() *FromGpuBox {
	box := new(FromGpuBox)
	Register(box)
	return box
}

func (box *FromGpuBox) Run() {
	LockCudaThread()
	str := cu.StreamCreate()
	defer str.Destroy()
	for {
		in := RecvGpu(box.Input)
		sendToHost(in, box.Output, str)
	}
}

func sendToHost(in GpuBlock, out []chan<- Block, stream cu.Stream) {
	buffer := Buffer()
	in.CopyDtoHAsync(buffer, stream)
	stream.Synchronize()
	RecycleGpu(in)
	Send(out, buffer)
}
