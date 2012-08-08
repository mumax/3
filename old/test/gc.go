package main

import (
	"fmt"
	. "nimble-cube/nc"
	"os"
)

func main() {

	n := 1
	InitSize(n, n, n)

	source := new(Source)
	sink := new(Sink)
	pass := new(Pass)
	Register(source, pass, sink)

	Connect(&sink.Input, &pass.Output)
	Connect(&pass.Input, &source.Output)
	WriteGraph("gc")

	go source.Run()
	go pass.Run()
	sink.Run(100)

	fmt.Println("NumAlloc:", NumAlloc())
	if NumAlloc() > 10 {
		os.Exit(1)
	}
}

type Source struct {
	Output []chan<- Block
}

func (box *Source) Run() {
	for {
		Send(box.Output, Buffer())
	}
}

type Sink struct {
	Input <-chan Block
}

func (box *Sink) Run(n int) {
	for i := 0; i < n; i++ {
		//log.Println("step", i)
		Recycle(Recv(box.Input))
	}
}

type Pass struct {
	Input  <-chan Block
	Output []chan<- Block
}

func (box *Pass) Run() {
	for {
		in := Recv(box.Input)
		out := Buffer()
		copy(out.List, in.List)
		Recycle(in)
		Send(box.Output, out)
	}
}
