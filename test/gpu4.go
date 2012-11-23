package main

import (
	"code.google.com/p/nimble-cube/cpu"
	"code.google.com/p/nimble-cube/gpu"
	"code.google.com/p/nimble-cube/gpu/conv"
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
		α          = 1
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
	B := conv.NewSymm2D("B", "T", mesh, mem, kernel, m).Output()

	exch := gpu.NewExchange6("Bex", m, Aex_red)
	nimble.Stack(exch)
	Bex := exch.Output()

	Beff := gpu.NewSum("Beff", B, Bex, Bsat, 1, mem).Output()

	tBox := gpu.NewLLGTorque("torque", m, Beff, α)
	nimble.Stack(tBox)
	torque := tBox.Output()

	dt := 200e-15
	solver := gpu.NewHeun(m, torque, mag.Gamma0, dt)

	every := 1000
	nimble.Autosave(B, every)
	nimble.Autosave(m, every)
	nimble.Autosave(Bex, every)
	nimble.Autosave(Beff, every)
	nimble.Autosave(torque, every)
	nimble.Autotable(m, every/10)

	nimble.RunStack()

	solver.Steps(5000)

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

	//	dt := 10e-15
	//	solver.SetDt(dt)
	//	tBox.SetAlpha(0.02)
	//	BeffBox.Madd(hext)
	//	D:=1e-9
	//	solver.Steps(D / dt)
}
