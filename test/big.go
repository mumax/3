package main

import (
	"code.google.com/p/mx3/gpu"
	"code.google.com/p/mx3/mag"
	"code.google.com/p/mx3/nimble"
	"fmt"
	"math"
)

func main() {
	nimble.Init()
	defer nimble.Cleanup()
	nimble.SetOD("gpu4.out")

	mem := nimble.GPUMemory

	const (
		a          = 8
		N0, N1, N2 = 1, 32 * a, 128 * a
		c          = 1e-9
		cx, cy, cz = c, c, c
		Bsat       = 800e3 * mag.Mu0
		Aex_red    = 13e-12 / (Bsat / mag.Mu0)
		α          = 1
	)

	mesh := nimble.NewMesh(N0, N1, N2, cx, cy, cz)
	fmt.Println("mesh:", mesh)

	m := nimble.MakeChanN(3, "m", "", mesh, mem, 0)
	M := gpu.Device3(m.ChanN().UnsafeData())
	M[0].Memset(float32(1 / math.Sqrt(3)))
	M[1].Memset(float32(1 / math.Sqrt(3)))
	M[2].Memset(float32(1 / math.Sqrt(3)))

	const acc = 1
	kernel := mag.BruteKernel(mesh, acc)
	B := gpu.NewConvolution("B", "T", mesh, mem, kernel, m).Output()
	Bex := gpu.NewExchange6("Bex", m, Aex_red).Output()

	BeffBox := gpu.NewSum("Beff", B, Bex, Bsat, 1, mem)
	Beff := BeffBox.Output()

	tBox := gpu.NewLLGTorque("torque", m, Beff, α)
	torque := tBox.Output()

	solver := gpu.NewHeun(m, torque, 10e-15, mag.Gamma0)
	solver.Maxerr = 2e-4
	solver.Mindt = 1e-15

	solver.Steps(100)
}

func sqr(x float32) float32 { return x * x }
