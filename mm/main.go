package mm

import (
	. "nimble-cube/nc"
)

func Main() {

	InitSize(1, 4, 8)

	// 1) make and connect boxes
	Register(new(LLGBox))
	Register(new(MeanFieldBox))

	solver := new(EulerBox)
	solver.dt = 0.01
	Register(solver)

	Register(NewConstBox(0.1), `Output:"alpha"`)

	//	avg := new(AverageBox)
	//	Connect1(avg, &avg.in, solver, &solver.mOut[X], "mx")
	//
	//	out := NewTableBox("mx.txt")
	//	ConnectManyFloat64(out, &out.input, avg, &avg.out, "<mx>")
	//	ConnectFloat64(out, &out.time, solver, &solver.time, "t")

	// 3) run boxes, no more should be created from now
	// -> let plumber do this.

	Start()

	// TODO: makearray
	m0 := [3][]float32{make([]float32, N()), make([]float32, N()), make([]float32, N())}
	Memset3(m0, Vector{0.1, 0.99, 0})

	// Solver box runs synchronous.
	// Could be async with return channel...
	for i := 0; i < 1000; i++ {
		solver.Run(m0, 10)
		//fmt.Println(m0[X][0], m0[Y][0], m0[Z][0])
	}

	// 4) tear-down and wait for boxes to finish
}
