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
		c0, c1, c2 = 2e-9, 4e-9, 4e-9
		Bsat       = 1100e3 * mag.Mu0
		Aex        = 13e-12
		alpha      = 0.2
		K1         = 0.5e6
	)

	mesh = data.NewMesh(N0, N1, N2, c0, c1, c2)
	M, Hd, Hex, Heff, Ha, torque := newVec(), newVec(), newVec(), newVec(), newVec(), newVec()

	cuda.Memset(M, 1, 1, 1)

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

	for i, by := range []float32{0, 10, 30, 100, 300} {
		By = by * 1e-3
		for solver.Time < float64(i+1)*1e-9 {
			solver.Step()
			if solver.NSteps%10 == 0 {
				save()
			}
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
