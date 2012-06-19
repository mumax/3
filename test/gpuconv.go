package main

import (
	. "nimble-cube/nc"
)

func main() {

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

type Source struct {
	Output []chan<- Block
}

func (box *Source) Run() {
	count := 0
	for {
		b := Buffer()
		for i := range b.List {
			b.List[i] = float32(count)
			count++
		}
		Send(box.Output, b)
	}
}

type Sink struct {
	Input <-chan Block
}

func (box *Sink) Run(n int) {
	count := 0
	for s := 0; s < n; s++ {
		in := Recv(box.Input)
		//Debug(in)
		for i := range in.List {
			if in.List[i] != float32(count) {
				Panic(in)
			}
			count++
		}
		Recycle(in)
	}
}

type GpuSource3 struct {
	Output [3][]chan<- GpuBlock
}

func (box *GpuSource3) Run() {
	for s:=0;s<NumWarp();s++{
		SendGpu(box.Output[0], GpuBuffer())
		SendGpu(box.Output[1], GpuBuffer())
		SendGpu(box.Output[2], GpuBuffer())
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
