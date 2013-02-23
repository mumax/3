// +build ignore

package main

import (
	"code.google.com/p/mx3/cuda"
	"code.google.com/p/mx3/data"
	"code.google.com/p/mx3/mag"
)

func main() {
	cuda.Init()

	const (
		N0, N1, N2 = 1, 32, 128
		S0, S1, S2 = 3e-9, 125e-9, 500e-9
		c0, c1, c2 = S0 / N0, S1 / N1, S2 / N2
		Bsat       = 800e3 * mag.Mu0
		Aex_red    = 13e-12 / (Bsat / mag.Mu0)
		α          = 1
	)

	mesh := data.NewMesh(N0, N1, N2, c0, c1, c2)
	m := cuda.NewSlice(3, mesh)
	cuda.Memset(m, 0, 1, 1)

	demag := cuda.NewDemag(mesh)
	hd := cuda.NewSlice(3, mesh)
	exch := cuda.NewExchange6(mesh, Aex_red)
	he := cuda.NewSlice(3, mesh)
	heff := cuda.NewSlice(3, mesh)
	torque := cuda.NewSlice(3, mesh)

	updateTorque := func(m *data.Slice) *data.Slice {
		demag.Exec(hd, m)
		exch.Exec(he, m)
		cuda.Madd2(heff, hd, he, Bsat, 1)
		cuda.LLGTorque(torque, m, heff, α)
		return torque
	}

	solver := cuda.NewHeun(m, updateTorque, 1e-15, mag.Gamma0)

	for solver.Time < 1e-9 {
		solver.Step()
	}
}
