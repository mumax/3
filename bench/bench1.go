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
