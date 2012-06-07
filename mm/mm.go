package mm

import (
	"fmt"
	. "nimble-cube/nc"
)

func Main() {

	// 0) initialize size, warp, etc
	InitSize(1, 4, 8)

	// 1) make and connect boxes
	torqueBox := new(LLGBox)
	hBox := new(MeanFieldBox)
	solver := new(EulerBox)
	solver.dt = 0.01
	alphaBox := NewConstBox(0.1)

	Connect3(hBox, &hBox.m, solver, &solver.mOut, "m")
	Connect3(torqueBox, &torqueBox.m, solver, &solver.mOut, "m")
	Connect3(torqueBox, &torqueBox.h, hBox, &hBox.h, "H")
	Connect(torqueBox, &torqueBox.alpha, alphaBox, &alphaBox.output, "alpha")
	Connect3(solver, &solver.torque, torqueBox, &(torqueBox.torque), "torque")
	Connect3(solver, &solver.mIn, solver, &solver.mOut, "m")

	avg := new(AverageBox)
	Connect(avg, &avg.in, solver, &solver.mOut[X], "mx")

	out := NewTableBox("mx.txt")
	ConnectFloat64(out, &out.input, avg, &avg.out, "<mx>")
	ConnectFloat64(out, &out.time, solver, &solver.time, "t")

	dot.Close() // how to automate?

	//	Probe3(&(solver.m), "m")
	//	Probe3(&(hBox.h), "h")
	//	Probe3(&(torqueBox.t), "t")

	// 3) run boxes, no more should be created from now
	// -> let plumber do this.
	go torqueBox.Run()
	go hBox.Run()
	go alphaBox.Run()
	go avg.Run()
	go out.Run()

	m0 := [3][]float32{make([]float32, N), make([]float32, N), make([]float32, N)}
	Memset3(m0, Vector{0.1, 0.99, 0})

	// Solver box runs synchronous.
	// Could be async with return channel...
	for i := 0; i < 1000; i++ {
		solver.Run(m0, 10)
		fmt.Println(m0[X][0], m0[Y][0], m0[Z][0])
	}

	// 4) tear-down and wait for boxes to finish
}

