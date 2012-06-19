package main

import (
	"fmt"
	"nimble-cube/mm"
	. "nimble-cube/nc"
	"os"
)

func main() {

	n := 1
	InitSize(n, n, n)

	source := new(Source)
	sink := new(Sink)
	sink2 := new(Sink)
	sink3 := new(Sink)
	c := mm.NewConstBox(1)

	Connect(&sink3.Input, &c.Output)

	Register(source, sink, sink2, sink3, c)
	AutoConnect(source, sink, sink2)
	WriteGraph("gc2")

	go source.Run()
	go sink2.Run(100)
	go sink3.Run(100)
	sink.Run(100)

	fmt.Println("NumAlloc:", NumAlloc)
	if NumAlloc > 10 {
		os.Exit(1)
	}
}

type Source struct {
	Output []chan<- Block "data"
}

func (box *Source) Run() {
	for {
		Send(box.Output, Buffer())
	}
}

type Sink struct {
	Input <-chan Block "data"
}

func (box *Sink) Run(n int) {
	for i := 0; i < n; i++ {
		//log.Println("step", i)
		Recycle(Recv(box.Input))
	}
}
