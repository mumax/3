package main

import (
	. "nimble-cube/nc"
	"log"
)

func main() {

	n := 1
	InitSize(n, n, n)

	source := new(Source)
	sink := new(Sink)

	AutoConnect(source, sink)
	WriteGraph()

	go source.Run()
	sink.Run(100)
}

type Source struct {
	Output []chan<- []float32 "data"
}

func (box *Source) Run() {
	for {
		Send(box.Output, Buffer())
	}
}

type Sink struct {
	Input <-chan []float32 "data"
}

func (box *Sink) Run(n int) {
	for i := 0; i < n; i++ {
		log.Println("step", i)
		Recycle(Recv(box.Input))
	}
}
