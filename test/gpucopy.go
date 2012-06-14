package main

import (
	. "nimble-cube/nc"
	"fmt"
	"os"
)

func main() {

	n := 4
	InitSize(n, n, n)

	source := new(Source)
	to := NewToGpuBox()
	gpusink := new(GpuSink)

	from := NewFromGpuBox()
	sink := new(Sink)

	Register(source, to, sink, gpusink)

	Connect(&to.Input, &source.Output)
	Connect(&from.Input, &to.Output)
	Connect(&gpusink.Input, &to.Output)
	Connect(&sink.Input, &from.Output)

	Vet(source, to, sink, gpusink)
	WriteGraph("gpucopy")

	GoRun(source, to, from, gpusink)
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
	count:=0
	for {
		b:=Buffer()
		for i:=range b{
			b[i]=float32(count)
			count++
		}		
		Send(box.Output,b )
	}
}

type Sink struct {
	Input <-chan []float32
}

func (box *Sink) Run(n int) {
	count:=0
	for s := 0; s < n; s++ {
		in:= Recv(box.Input)
		//Debug(in)
		for i:=range in{
			if in[i] != float32(count){Panic(in)}
		count++
		}	
		Recycle(in)
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

func (box *GpuSink) Run() {
	for {
		RecycleGpu(RecvGpu(box.Input))
	}
}
