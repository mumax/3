package main

import (
	"flag"
	"fmt"
	. "nimble-cube/mm"
	. "nimble-cube/nc"
	"strconv"
	"time"
)

func main() {

	n, err := strconv.Atoi(flag.Arg(0))
	CheckIO(err)
	InitSize(n, n, n)

	torque := new(LLGBox)
	heff := new(MeanFieldBox)
	alpha := NewConstBox(0.1)
	solver := new(EulerBox)
	//solver.dt = 0.01

	AutoConnect(torque, alpha, heff, solver, alpha)
	AutoRun()
	WriteGraph()

	// TODO: makearray
	m0 := [3][]float32{make([]float32, N()), make([]float32, N()), make([]float32, N())}
	Memset3(m0, Vector{0.1, 0.99, 0})

	start := time.Now()
	solver.Run(m0, 1000)
	duration := time.Now().Sub(start)
	fmt.Println("# N	warp	ms/step")
	fmt.Println(n*n*n, "\t", WarpLen(), "\t", float64(duration)/(1e9))

}


type Source struct {
	Output []chan<- []float32
}

func (box *Source) Run() {
	for {
		Send(box.Output, Buffer())
	}
}

type Sink struct {
	Input <-chan []float32
}

func (box *Sink) Run() {
	for {
		Recycle(Recv(box.Input))
	}
}

type Source3 struct {
	Output [3][]chan<- []float32
}

func (box *Source3) Run() {
	for {
		Send3(box.Output, Buffer3())
	}
}

type Sink3 struct {
	Input <-chan [3][]float32
}

func (box *Sink3) Run() {
	for {
		Recycle3(Recv3(box.Input))
	}
}
