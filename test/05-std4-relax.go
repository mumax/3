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
		S0, S1, S2 = 3e-9, 125e-9, 500e-9
		c0, c1, c2 = S0 / N0, S1 / N1, S2 / N2
		Bsat       = 800e3 * mag.Mu0
		Aex        = 13e-12
		α          = 1
	)

	mesh = data.NewMesh(N0, N1, N2, c0, c1, c2)
	m, Beff, torque := newVec(), newVec(), newVec()
	vol := data.NilSlice(1, mesh)

	cuda.Memset(m, 0, 1, 1)

	demag := cuda.NewDemag(mesh)

	updateTorque := func(m *data.Slice, t float64) *data.Slice {
		demag.Exec(Beff, m, vol, Bsat)
		cuda.AddExchange(Beff, m, Aex, Bsat)
		cuda.LLGTorque(torque, m, Beff, α)
		return torque
	}

	solver := cuda.NewHeun(m, updateTorque, cuda.Normalize, 1e-15, mag.Gamma0)

	solver.Advance(2e-9)
	mx, my, mz := m.Comp(0), m.Comp(1), m.Comp(2)
	N := float32(mesh.NCell())
	avgx, avgy, avgz := cuda.Sum(mx)/N, cuda.Sum(my)/N, cuda.Sum(mz)/N
	fmt.Println(avgx, avgy, avgz)
	expect(avgx, 0)
	expect(avgy, 0.125)
	expect(avgz, 0.967)
	fmt.Println("OK")
}

func expect(have, want float32) {
	if abs(have-want) > 1e-2 {
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
