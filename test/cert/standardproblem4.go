// +build ignore

package main

import (
	"code.google.com/p/mx3/cuda"
	"code.google.com/p/mx3/data"
	"code.google.com/p/mx3/engine"
	"code.google.com/p/mx3/mag"
	"fmt"
	"log"
)

var mesh *data.Mesh

func main() {
	engine.Init()
	cuda.Init()
	cuda.LockThread()

	const (
		N0, N1, N2 = 1, 32, 128
		N          = N0 * N1 * N2
		S0, S1, S2 = 3e-9, 125e-9, 500e-9
		c0, c1, c2 = S0 / N0, S1 / N1, S2 / N2
		Bsat       = 800e3 * mag.Mu0
		Aex        = 13e-12
	)
	alpha := float32(1.0)

	mesh = data.NewMesh(N0, N1, N2, c0, c1, c2)
	M, Hd, Hex, Heff, torque := newVec(), newVec(), newVec(), newVec(), newVec()

	cuda.Memset(M, 0, 1, 1)

	demag := cuda.NewDemag(mesh)

	var Bx, By, Bz float32 = 0, 0, 0

	updateTorque := func(m *data.Slice) *data.Slice {
		demag.Exec(Hd, m)
		cuda.Exchange(Hex, m, Aex)
		cuda.Madd2(Heff, Hd, Hex, 1, 1)
		cuda.AddConst(Heff, Bx, By, Bz)
		cuda.LLGTorque(torque, m, Heff, alpha)
		return torque
	}

	norm := func(m *data.Slice) {
		cuda.Normalize(m, nil, Bsat)
	}
	norm(M)

	solver := cuda.NewHeun(M, updateTorque, norm, 1e-15, mag.Gamma0)

	for solver.Time < 3e-9 {
		solver.Step()
	}
	mx, my, mz := M.Comp(0), M.Comp(1), M.Comp(2)
	//avgx, avgy, avgz := cuda.Sum(mx)/(N*Bsat), cuda.Sum(my)/(N*Bsat), cuda.Sum(mz)/(N*Bsat)
	////fmt.Println(avgx, avgy, avgz)
	//expect(avgx, 0)
	//expect(avgy, 0.12358)
	//expect(avgz, 0.95588)
	//fmt.Println("relax OK")

	alpha = 0.02
	By = 4.3E-3
	Bz = -24.6E-3

	solver.Time = 0
	for solver.Time < 1e-9 {
		solver.Step()
		if solver.NSteps%10 == 0 {
			avgx, avgy, avgz := cuda.Sum(mx)/(N*Bsat), cuda.Sum(my)/(N*Bsat), cuda.Sum(mz)/(N*Bsat)
			fmt.Println(solver.Time, avgx, avgy, avgz)
			//if solver.NSteps % 100 == 0{
			//	data.MustWriteFile(fmt.Sprintf("m%06d.dump", solver.NSteps), M.HostCopy(), solver.Time)
			//}
		}
	}
}

func expect(have, want float32) {
	if abs(have-want) > 1e-3 {
		log.Fatalln("have:", have, "want:", want)
	}
}

func abs(x float32) float32 {
	if x < 0 {
		return -x
	}
	return x
}

func newVec() *data.Slice {
	return cuda.NewSlice(3, mesh)
}
