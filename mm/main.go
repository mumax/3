package mm

import (
	"log"
	. "nimble-cube/nc"
)

func Main() {

	InitSize(1, 4, 8)

	torque := new(LLGBox)
	heff := new(MeanFieldBox)
	alpha := NewConstBox(0.1)
	solver := new(EulerBox)
	solver.dt = 0.01
	avg := new(Average3Box)
	table := NewTableBox("m.txt")

	AutoConnect(torque, alpha, heff, avg, table, solver)
	AutoRun()
	WriteGraph()

	// TODO: makearray
	m0 := [3][]float32{make([]float32, N()), make([]float32, N()), make([]float32, N())}
	Memset3(m0, Vector{0.1, 0.99, 0})

	// Solver box runs synchronous.
	// Could be async with return channel...
	//for i := 0; i < 1000; i++ {
	log.Println("start running")
	solver.Run(m0, 10000)
	log.Println("done running")
	//fmt.Println(m0[X][0], m0[Y][0], m0[Z][0])
	//}

}
