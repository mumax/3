package main

import (
	. "nimble-cube/nc"
)

func main() {

	MAX_WARPLEN = 32
	InitSize(1, 8, 16)

	m := new(GpuSource3)
	conv := NewGpuConvBox()
	Register(m, conv)

	Connect(&conv.M[X], &m.Output[X])
	Connect(&conv.M[Y], &m.Output[Y])
	Connect(&conv.M[Z], &m.Output[Z])

	Vet(m, conv)
	WriteGraph("gpuconv")

	GoRun(m)
	conv.Run() // once

}

type GpuSource3 struct {
	Output [3][]chan<- GpuBlock
}

func (box *GpuSource3) Run() {
	LockCudaCtx()
	for s := 0; s < NumWarp(); s++ {
		buf := GpuBuffer()
		buf.Memset(1)
		buf.Set(0, 0, 0, 2)
		SendGpu(box.Output[0], buf)

		buf = GpuBuffer()
		buf.Memset(2)
		buf.Set(0, 0, 0, 3)
		SendGpu(box.Output[1], buf)

		buf = GpuBuffer()
		buf.Memset(3)
		buf.Set(0, 0, 0, 4)
		SendGpu(box.Output[2], buf)
	}
}

type GpuSink struct {
	Input <-chan GpuBlock
}

func (box *GpuSink) Run() {
	for {
		RecycleGpu(RecvGpu(box.Input))
	}
}
