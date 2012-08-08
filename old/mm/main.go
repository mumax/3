package mm

import (
	"flag"
	"log"
	. "nimble-cube/nc"
	"strconv"
)

func Main() {

	N0 := intFlag(0)
	N1 := intFlag(1)
	N2 := intFlag(2)

	InitSize(N0, N1, N2)

	torque := new(LLGBox)
	heff := new(MeanFieldBox)
	alpha := NewConstBox(0.1)
	solver := new(EulerBox)
	solver.dt = 0.01
	avg := NewAverage3Box()
	Register(avg)
	table := NewTableBox("m.txt")

	Connect(&avg.Input, &solver.MOut)
	Connect(&table.Input, &avg.Output[X])
	Connect(&torque.Alpha, &alpha.Output)

	AutoConnect(torque, alpha, heff, avg, table, solver)
	AutoRun()
	WriteGraph("mm")

	// TODO: makearray
	m0 := [3]Block{MakeBlock(Size()), MakeBlock(Size()), MakeBlock(Size())}
	Memset(m0[X].List, 0.1)
	Memset(m0[X].List, 0.99)
	Memset(m0[X].List, 0)

	// Solver box runs synchronous.
	// Could be async with return channel...
	//for i := 0; i < 1000; i++ {
	log.Println("start running")
	solver.Run(m0, 10000)
	log.Println("done running")
	//fmt.Println(m0[X][0], m0[Y][0], m0[Z][0])
	//}

}

func intFlag(i int) int {
	v, err := strconv.Atoi(flag.Arg(i))
	PanicErr(err)
	return v
}
