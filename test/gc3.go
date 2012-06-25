package main

import (
	"fmt"
	. "nimble-cube/nc"
	"os"
)

func main() {

	n := 1
	InitSize(n, n, n)

	source := new(Source3)
	sink := new(Sink3)
	//sink2 := new(Sink3)

	AutoConnect(source, sink) //, sink2)
	WriteGraph("gc3")

	go source.Run()
	//go sink2.Run(100)
	sink.Run(100)

	fmt.Println("NumAlloc:", NumAlloc())
	if NumAlloc() > 10 {
		os.Exit(1)
	}
}

type Source3 struct {
	Output [3][]chan<- Block "data"
}

func (box *Source3) Run() {
	for {
		Send3(box.Output, Buffer3())
	}
}

type Sink3 struct {
	Input [3]<-chan Block "data"
}

func (box *Sink3) Run(n int) {
	for i := 0; i < n; i++ {
		Recycle3(Recv3(box.Input))
	}
}
