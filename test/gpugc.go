package main

import (
	. "nimble-cube/nc"
	"fmt"
	"log"
	"os"
)

func main() {

	n := 1
	InitSize(n, n, n)

	source := new(Source)
	sink := new(Sink)
	Register(source, sink)

	Connect(&sink.Input, &source.Output)
	WriteGraph("gc")

	GoRun(source)
	sink.Run(100)

	fmt.Println("NumGpuAlloc:", NumGpuAlloc)
	if NumAlloc > 10 {
		os.Exit(1)
	}
}

type Source struct {
	Output []chan<- GpuFloats
}

func (box *Source) Run() {
	for {
		SendGpu(box.Output, GpuBuffer())
	}
}

type Sink struct {
	Input <-chan GpuFloats
}

func (box *Sink) Run(n int) {
	for i := 0; i < n; i++ {
		log.Println("step", i)
		RecycleGpu(RecvGpu(box.Input))
	}
}

