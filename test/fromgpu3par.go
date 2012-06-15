package main

import (
	"fmt"
	. "nimble-cube/nc"
	"os"
)

func main() {

	MAX_WARP = 4
	n := 4
	InitSize(n, n, n)

	sourceX := new(GpuSource)
	sourceY := new(GpuSource)
	sourceZ := new(GpuSource)
	from := NewFromGpu3Par()
	sink := new(Sink3)

	Register(sourceX, sourceY, sourceZ, from, sink)

	Connect(&from.Input[X], &sourceX.Output)
	Connect(&from.Input[Y], &sourceY.Output)
	Connect(&from.Input[Z], &sourceZ.Output)
	Connect(&sink.Input, &from.Output)

	Vet(sourceX, sourceY, sourceZ, from, sink)
	WriteGraph("fromgpu3par")

	GoRun(from, sourceX, sourceY, sourceZ)
	sink.Run(50)

	fmt.Println("NumAlloc:", NumAlloc)
	if NumAlloc > 30 {
		os.Exit(1)
	}
	fmt.Println("NumGpuAlloc:", NumGpuAlloc)
	if NumGpuAlloc > 30 {
		os.Exit(1)
	}
}




type Sink3 struct {
	Input [3]<-chan []float32
}

func (box *Sink3) Run(n int) {
	Log("sinking", n)
	for s := 0; s < n; s++ {
		Log("step", s)
		in := Recv3(box.Input)
		Recycle3(in)
	}
}

type GpuSource struct {
	Output []chan<- GpuFloats
}

func (box *GpuSource) Run() {
	for {
		SendGpu(box.Output, GpuBuffer())
	}
}

