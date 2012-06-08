package mm

import (
	. "nimble-cube/nc"
)

func Main() {

	InitSize(1, 4, 8)

	// 1) make and connect boxes
	Register(new(LLGBox))
	Register(new(MeanFieldBox))
	Register(NewConstBox(0.1)) //, `Output:"alpha"`)

	solver := new(EulerBox)
	solver.dt = 0.01
	Register(solver)

	avg := new(AverageBox)
	Register(avg)
	ManualConnect(avg, &avg.Input, solver, &solver.MOut[X], "mx")

	//Output("m.x", "mx.txt")

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
