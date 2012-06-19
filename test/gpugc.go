package main

import (
	"fmt"
	. "nimble-cube/nc"
	"os"
)

func main() {

	n := 1
	InitSize(n, n, n)

	source := new(GpuSource)
	sink := new(GpuSink)
	sink2 := new(GpuSink)
	Register(source, sink, sink2)

	Connect(&sink.Input, &source.Output)
	Connect(&sink2.Input, &source.Output)
	WriteGraph("gpugc")

	GoRun(source)
	go sink2.Run(100)
	sink.Run(100)

	fmt.Println("NumGpuAlloc:", NumGpuAlloc)
	if NumAlloc > 10 {
		os.Exit(1)
	}
}

type GpuSource struct {
	Output []chan<- GpuBlock
}

func (box *GpuSource) Run() {
	SetCudaCtx()
	for {
		SendGpu(box.Output, GpuBuffer())
	}
}

type GpuSink struct {
	Input <-chan GpuBlock
}

func (box *GpuSink) Run(n int) {
	SetCudaCtx()
	for i := 0; i < n; i++ {
		//log.Println("step", i)
		RecycleGpu(RecvGpu(box.Input))
	}
}
