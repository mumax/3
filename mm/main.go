package mm

import (
	. "nimble-cube/nc"
	"log"
)

func Main() {

	InitSize(1, 4, 8)

	// 1) make and connect boxes
	torque := new(LLGBox)
	heff := new(MeanFieldBox)
	alpha := NewConstBox(0.1)

	Connect(&torque.Alpha, &alpha.Output)

	solver := new(EulerBox)
	solver.dt = 0.01

	Connect(&solver.MIn, &solver.MOut)
	Connect(&solver.Torque, &torque.Torque)

	Connect(&alpha.Time, &solver.Time)

	Connect(&torque.H, &heff.H)
	Connect(&torque.M, &solver.MOut)

	Connect(&heff.M, &solver.MOut)

	Register(torque, alpha, solver, heff)
	WriteGraph()

	GoRun(torque, alpha, heff)

	//RegisterBox(NewAverage3Box("m"))
	//RegisterBox(NewTableBox("m.txt", "<m>.x"))
	//Output("m.x", "mx.txt")

	//Start()

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
