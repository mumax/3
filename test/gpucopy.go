package main

import (
	. "nimble-cube/nc"
	//"runtime"
	"fmt"
	"log"
	"os"
)

func main() {

	n := 4
	InitSize(n, n, n)

	source := new(Source)
	to := NewToGpuBox()
	//gpusink := new(GpuSink)

	from := NewFromGpuBox()
	sink := new(Sink)

	Register(source, to, sink)

	Connect(&to.Input, &source.Output)
	Connect(&from.Input, &to.Output)
	Connect(&sink.Input, &from.Output)

	Vet(source, to, sink)
	WriteGraph("gpucopy")

	//GoRun(source, to)
	go func(){SetCudaCtx(); source.Run()}()
	go func(){SetCudaCtx(); to.Run()}()
	go func(){SetCudaCtx(); from.Run()}()
	sink.Run(100)

	fmt.Println("NumAlloc:", NumAlloc)
	if NumAlloc > 10 {
		os.Exit(1)
	}
	fmt.Println("NumGpuAlloc:", NumGpuAlloc)
	if NumGpuAlloc > 10 {
		os.Exit(1)
	}
}

type Source struct {
	Output []chan<- []float32
}

func (box *Source) Run() {
	Debug("run source")
	for {
		Debug("sending")
		Send(box.Output, Buffer())
		Debug("sent")
	}
}

type Sink struct {
	Input <-chan []float32
}

func (box *Sink) Run(n int) {
	for i := 0; i < n; i++ {
		log.Println("step", i)
		Recycle(Recv(box.Input))
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

type GpuSink struct {
	Input <-chan GpuFloats
}

func (box *GpuSink) Run(n int) {
	for i := 0; i < n; i++ {
		log.Println("step", i)
		RecycleGpu(RecvGpu(box.Input))
	}
}

