// +build ignore

package main

import (
	"code.google.com/p/mx3/cuda"
	"code.google.com/p/mx3/data"
	"code.google.com/p/mx3/engine"
	"code.google.com/p/mx3/mag"
)

var mesh *data.Mesh

func main() {
	engine.Init()
	cuda.Init()
	cuda.LockThread()

	const (
		a          = 2
		N0, N1, N2 = 1 * a, 32 * a, 128 * a
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

	updateTorque := func(m *data.Slice, t float64) *data.Slice {
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

	solver.Steps(10)
}

func newVec() *data.Slice {
	return cuda.NewSlice(3, mesh)
}
