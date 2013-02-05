package main

import (
	"code.google.com/p/mx3/cpu"
	"code.google.com/p/mx3/gpu"
	"code.google.com/p/mx3/gpu/conv"
	"code.google.com/p/mx3/mag"
	"code.google.com/p/mx3/nimble"
	"fmt"
	"os"
)

func main() {
	nimble.Init()
	defer nimble.Cleanup()
	nimble.SetOD("gpullgtorque.out")

	N0, N1, N2 := 1, 32, 128
	cx, cy, cz := 3e-9, 3.125e-9, 3.125e-9
	mesh := nimble.NewMesh(N0, N1, N2, cx, cy, cz)
	fmt.Println("mesh:", mesh)

	m := nimble.MakeChanN(3, "m", "", mesh, nimble.UnifiedMemory, 0)
	fmt.Println("m:", m)

	acc := 8
	kernel := mag.BruteKernel(mesh, acc)
	B := conv.NewSymm2D("B", "T", mesh, nimble.UnifiedMemory, kernel, m).Output()

	const (
		Bsat = 1.0053
		aex  = mag.Mu0 * 13e-12 / Bsat
		α    = 1
	)
	//exch := cpu.NewExchange6("Bex", "T", nimble.UnifiedMemory, m.NewReader(), aex)
	exch := gpu.NewExchange6("Bex", m, aex)
	nimble.Stack(exch)
	Bex := exch.Output()

	//	heff := MakeChan3("Heff", "", mesh)
	Beff := gpu.NewSum("Beff", B, Bex, Bsat, 1, nimble.UnifiedMemory).Output()

	tBox := gpu.NewLLGTorque("torque", m, Beff, α)
	nimble.Stack(tBox)
	torque := tBox.Output()

	const dt = 100e-15

	solver := cpu.NewEuler(m, torque.NewReader(), mag.Gamma0, dt)

	M := cpu.Host(m.ChanN().UnsafeData())
	for i := range M[2] {
		M[2][i] = 1
		M[1][i] = 0.1
	}

	every := 100
	nimble.Autosave(B, every)
	nimble.Autosave(m, every)
	nimble.Autosave(Bex, every)
	nimble.Autosave(Beff, every)
	nimble.Autosave(torque, every)
	nimble.Autotable(m, every)

	nimble.RunStack()

	solver.Steps(100)
	res := cpu.Host(m.ChanN().UnsafeData())
	got := [3]float32{res[0][0], res[1][0], res[2][0]}
	expect := [3]float32{-0.033120323, 0.20761484, 0.9776498}
	solver.Steps(10000)
	fmt.Println("result:", got)
	if got != expect {
		fmt.Println("expected:", expect)
		os.Exit(2)
	} else {
		fmt.Println("OK")
	}
}
