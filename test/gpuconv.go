package main

import (
	. "nimble-cube/nc"
)

func main() {

	MAX_WARPLEN = 256
	InitSize(4, 8, 16)

	m := new(GpuSource3)
	conv := NewConvBox()
	kern := NewKernelBox()
	Register(m, conv, kern)

	Connect(&conv.M[X], &m.Output[X])
	Connect(&conv.M[Y], &m.Output[Y])
	Connect(&conv.M[Z], &m.Output[Z])

	to := make([]*ToGpuBox, 6)
	for i := range to {
		to[i] = NewToGpuBox()
	}

	Connect(&conv.Kii, &to[0].Output)
	Connect(&conv.Kjj, &to[1].Output)
	Connect(&conv.Kkk, &to[2].Output)
	Connect(&conv.Kjk, &to[3].Output)
	Connect(&conv.Kik, &to[4].Output)
	Connect(&conv.Kij, &to[5].Output)

	Connect(&to[0].Input, &kern.Kii)
	Connect(&to[1].Input, &kern.Kjj)
	Connect(&to[2].Input, &kern.Kkk)
	Connect(&to[3].Input, &kern.Kjk)
	Connect(&to[4].Input, &kern.Kik)
	Connect(&to[5].Input, &kern.Kij)

	WriteGraph("gpuconv")

	GoRun(m, kern)
	for i := range to {
		GoRun(to[i])
	}
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
