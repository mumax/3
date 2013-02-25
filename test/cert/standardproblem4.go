// +build ignore

package main

import (
	"code.google.com/p/mx3/cuda"
	"code.google.com/p/mx3/data"
	"code.google.com/p/mx3/engine"
	"code.google.com/p/mx3/mag"
	"fmt"
)

var mesh *data.Mesh

func main() {
	engine.Init()
	cuda.Init()
	cuda.LockThread()

	const (
		N0, N1, N2 = 1, 32, 128
		S0, S1, S2 = 3e-9, 125e-9, 500e-9
		c0, c1, c2 = S0 / N0, S1 / N1, S2 / N2
		Bsat       = 800e3 * mag.Mu0
		Aex        = 13e-12
		α          = 1
	)

	mesh = data.NewMesh(N0, N1, N2, c0, c1, c2)
	M, Hd, Hex, Heff, torque := newVec(), newVec(), newVec(), newVec(), newVec()

	cuda.Memset(M, 0, 1, 1)

	demag := cuda.NewDemag(mesh)

	updateTorque := func(m *data.Slice) *data.Slice {
		demag.Exec(Hd, m)
		cuda.Exchange(Hex, m, Aex)
		cuda.Madd2(Heff, Hd, Hex, Bsat, 1)
		cuda.LLGTorque(torque, m, Heff, α)
		return torque
	}

	solver := cuda.NewHeun(M, updateTorque, 1e-15, mag.Gamma0)

	mx, my, mz := M.Comp(0), M.Comp(1), M.Comp(2)
	N := float32(mesh.NCell()) * Bsat
	for solver.Time < 2e-9 {
		if solver.NSteps%10 == 0 {
			fmt.Println(solver.Time, cuda.Sum(mx)/N, cuda.Sum(my)/N, cuda.Sum(mz)/N)
		}
		solver.Step()
	}

	data.MustWriteFile("m.dump", M.HostCopy(), 0)
}

func newVec() *data.Slice {
	return cuda.NewSlice(3, mesh)
}
