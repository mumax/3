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
		a          = 1
		N0, N1, N2 = 1 * a, 32 * a, 128 * a
		N          = N0 * N1 * N2
		S0, S1, S2 = 3e-9, 125e-9, 500e-9
		c0, c1, c2 = S0 / N0, S1 / N1, S2 / N2
		SCALE      = 5.                      // scale Bsat by this amount, compensate with mask
		Bsat       = 800e3 * mag.Mu0 * SCALE // !
		Aex        = 13e-12
		α          = 1
	)

	mesh = data.NewMesh(N0, N1, N2, c0, c1, c2)
	M, Hd, Hex, Heff, torque := newVec(), newVec(), newVec(), newVec(), newVec()

	cuda.Memset(M, 0, 1, 1)

	demag := cuda.NewDemag(mesh)

	updateTorque := func(m *data.Slice, t float64) *data.Slice {
		demag.Exec(Hd, m)
		cuda.Exchange(Hex, m, Aex)
		cuda.Madd2(Heff, Hd, Hex, 1, 1)
		cuda.LLGTorque(torque, m, Heff, α)
		return torque
	}

	mask := cuda.NewSlice(1, mesh)
	cuda.Memset(mask, 1./SCALE)
	norm := func(m *data.Slice) {
		cuda.Normalize(m, mask, Bsat)
	}

	solver := cuda.NewHeun(M, updateTorque, norm, 1e-15, mag.Gamma0)

	for solver.Time < 5e-9 {
		solver.Step()
	}
	cuda.Normalize(M, nil, 1)
	mx, my, mz := M.Comp(0), M.Comp(1), M.Comp(2)
	avgx, avgy, avgz := cuda.Sum(mx)/N, cuda.Sum(my)/N, cuda.Sum(mz)/N
	fmt.Println(avgx, avgy, avgz)

	data.MustWriteFile("m.dump", M.HostCopy(), 0)
	expect(avgx, 0)
	expect(avgy, 0.125)
	expect(avgz, 0.967)
	fmt.Println("OK")
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
