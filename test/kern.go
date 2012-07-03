package main

import (
	. "nimble-cube/nc"
)

func main() {

	MAX_WARPLEN = 256
	InitSize(4, 8, 16)

	kern := NewKernelBox()
	sink := new(Sink)

	Connect(&sink.Input, &kern.FFTKernel)
	go kern.Run()
	sink.Run() // once
}

type Sink struct {
	Input <-chan Block
}

func (box *Sink) Run() {
	for s := 0; s < NumWarp(); s++ {
		in := Recv(box.Input)
		Debug(in)
	}
}
