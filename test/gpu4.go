package main

import (
	"code.google.com/p/nimble-cube/cpu"
	"code.google.com/p/nimble-cube/gpu"
	"code.google.com/p/nimble-cube/mag"
	"code.google.com/p/nimble-cube/nimble"
	"fmt"
	"os"
)

// Standard problem 4 on GPU
func main() {
	nimble.Init()
	defer nimble.Cleanup()
	nimble.SetOD("gpu4.out")

	mem := nimble.UnifiedMemory

	const (
		N0, N1, N2 = 1, 32, 128
		cx, cy, cz = 3e-9, 3.125e-9, 3.125e-9
		Bsat       = 1.0053
		Aex_red    = mag.Mu0 * 13e-12 / Bsat
		α          = 0.02
		dt = 200e-15
	)

	mesh := nimble.NewMesh(N0, N1, N2, cx, cy, cz)
	fmt.Println("mesh:", mesh)

	m := nimble.MakeChanN(3, "m", "", mesh, mem, 0)
	M := cpu.Host(m.ChanN().UnsafeData())
	for i := range M[2] {
		M[2][i] = 1
		M[1][i] = 0.1
	}

	acc := 10
	kernel := mag.BruteKernel(mesh, acc)
	B := gpu.NewConvolution("B", "T", mesh, mem, kernel, m).Output()

	exch := gpu.NewExchange6("Bex", m, Aex_red)
	nimble.Stack(exch)
	Bex := exch.Output()

	BeffBox := gpu.NewSum("Beff", B, Bex, Bsat, 1, mem)
	Beff := BeffBox.Output()

	tBox := gpu.NewLLGTorque("torque", m, Beff, α)
	nimble.Stack(tBox)
	torque := tBox.Output()

	solver := gpu.NewHeun(m, torque, 0, mag.Gamma0)
	solver.SetDt(200e-15)

	every := 100
	nimble.Autosave(m, every)
	nimble.Autotable(m, every/10)

	D := 1e-9
	solver.Steps(int(D / dt))

	res := cpu.Host(m.ChanN().UnsafeData())
	got := [3]float32{res[0][0], res[1][0], res[2][0]}
	expect := [3]float32{1.090642e-06, 0.6730072, 0.739636}
	fmt.Println("result:", got)
	if got != expect {
		fmt.Println("expected:", expect)
		os.Exit(2)
	} else {
		fmt.Println("OK")
	}

	const (
		Bx = -24.6E-3
		By = 4.3E-3
		Bz = 0
	)
	Bext := gpu.RunConst("Bext", "T", mesh, mem, []float64{Bz, By, Bx})
	BeffBox.MAdd(Bext, 1)
	solver.Steps(int(D / dt))
}
