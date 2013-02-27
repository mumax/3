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
		N0, N1, N2 = 1, 64, 64
		N          = N0 * N1 * N2
		c0, c1, c2 = 3e-9, 2e-9, 2e-9
		Bsat       = 800e3 * mag.Mu0
		Aex        = 13e-12
		alpha      = 1
		K1         = 5000
	)

	mesh = data.NewMesh(N0, N1, N2, c0, c1, c2)
	M, Hd, Hex, Heff, Ha, torque := newVec(), newVec(), newVec(), newVec(), newVec(), newVec()

	cuda.Memset(M, 0, 1, -0.01)

	demag := cuda.NewDemag(mesh)

	var Bx, By, Bz float32 = 0, 0, 0
	var Kx, Ky, Kz float32 = 0, 0, K1

	updateTorque := func(m *data.Slice, t float64) *data.Slice {
		demag.Exec(Hd, m)
		cuda.Exchange(Hex, m, Aex)
		cuda.UniaxialAnisotropy(Ha, m, Kx, Ky, Kz)
		cuda.Madd3(Heff, Hd, Hex, Ha, 1, 1, 1)
		cuda.AddConst(Heff, Bx, By, Bz)
		cuda.LLGTorque(torque, m, Heff, alpha)
		return torque
	}

	norm := func(m *data.Slice) {
		cuda.Normalize(m, nil, Bsat)
	}
	norm(M)

	solver := cuda.NewHeun(M, updateTorque, norm, 1e-15, mag.Gamma0)

	save := func() {
		mx, my, mz := M.Comp(0), M.Comp(1), M.Comp(2)
		avgx, avgy, avgz := cuda.Sum(mx)/(N*Bsat), cuda.Sum(my)/(N*Bsat), cuda.Sum(mz)/(N*Bsat)
		fmt.Println(solver.Time, avgx, avgy, avgz)
	}

	for solver.Time < 1e-9 {
		solver.Step()
		if solver.NSteps%10 == 0 {
			save()
		}
	}

	By = 10e-3
	Bz = 10e-3

	for solver.Time < 2e-9 {
		solver.Step()
		if solver.NSteps%10 == 0 {
			save()
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
