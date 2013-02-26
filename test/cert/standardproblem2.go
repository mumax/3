// +build ignore

package main

import (
	"code.google.com/p/mx3/cuda"
	"code.google.com/p/mx3/data"
	"code.google.com/p/mx3/engine"
	"code.google.com/p/mx3/mag"
	"code.google.com/p/mx3/util"
	"fmt"
	"log"
	"math"
)

var mesh *data.Mesh

// Run with d/lex as command line arg.
func main() {
	engine.Init()
	cuda.Init()
	cuda.LockThread()

	I := util.FloatArg(0)
	Msat := 800e3 //* 2
	Bsat := float32(Msat * mag.Mu0)
	Aex := 13e-12
	alpha := float32(1.0)
	lex := math.Sqrt(Aex / (0.5 * mag.Mu0 * Msat * Msat))
	d := I * lex
	S0, S1, S2 := 0.1*d, d, 5*d

	N0, N1, N2 := 1, 1, 1
	for S1/float64(N1) > 0.75*lex {
		N1 *= 2
	}
	for S2/float64(N2) > 0.75*lex {
		N2 *= 2
	}
	N := float32(N0 * N1 * N2)

	mesh = data.NewMesh(N0, N1, N2, S0/float64(N0), S1/float64(N1), S2/float64(N2))
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

	solver.Steps(1000)
	solver.Relax(1e-6, 1e6)
	solver.Steps(1000)
	mx, my, mz := M.Comp(0), M.Comp(1), M.Comp(2)
	avgx, avgy, avgz := cuda.Sum(mx)/(N*Bsat), cuda.Sum(my)/(N*Bsat), cuda.Sum(mz)/(N*Bsat)
	fmt.Println(I, avgx, avgy, avgz)
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
