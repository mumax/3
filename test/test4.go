package main

import (
	"code.google.com/p/mx3/cpu"
	"code.google.com/p/mx3/gpu"
	"code.google.com/p/mx3/mag"
	"code.google.com/p/mx3/nimble"
	"code.google.com/p/mx3/uni"
	"fmt"
	"os"
)

func main() {
	nimble.Init()
	defer nimble.Cleanup()
	nimble.SetOD("test4.out")

	N0, N1, N2 := 1, 32, 128
	cx, cy, cz := 3e-9, 3.125e-9, 3.125e-9
	mesh := nimble.NewMesh(N0, N1, N2, cx, cy, cz)
	fmt.Println("mesh:", mesh)

	m := nimble.MakeChanN(3, "m", "", mesh, nimble.UnifiedMemory, 0)
	fmt.Println("m:", m)

	acc := 8
	kernel := mag.BruteKernel(mesh, acc)
	B := gpu.NewConvolution("B", "T", mesh, nimble.UnifiedMemory, kernel, m).Output()

	const Bsat = 1.0053
	const aex = mag.Mu0 * 13e-12 / Bsat
	exch := cpu.NewExchange6("Bex", "T", nimble.UnifiedMemory, m.NewReader(), aex)
	Bex := exch.Output()

	Beff := cpu.NewSum("Beff", B, Bex, Bsat, 1, nimble.UnifiedMemory).Output()

	const alpha = 1
	tbox := cpu.NewLLGTorque("torque", m, Beff, alpha)
	nimble.Stack(tbox)
	torque := tbox.Output()

	const dt = 100e-15

	solver := cpu.NewEuler(m, torque.NewReader(), mag.Gamma0, dt)

	M := cpu.Host(m.ChanN().UnsafeData())
	for i := range M[2] {
		M[2][i] = 1
		M[1][i] = 0.1
	}

	every := 100
	uni.Autosave(B, every, cpu.CPUDevice)
	uni.Autosave(m, every, cpu.CPUDevice)
	uni.Autosave(Bex, every, cpu.CPUDevice)
	uni.Autosave(Beff, every, cpu.CPUDevice)
	uni.Autosave(torque, every, cpu.CPUDevice)
	uni.Autotable(m, every, cpu.CPUDevice)

	nimble.RunStack()

	solver.Steps(100)
	res := cpu.Host(m.ChanN().UnsafeData())
	got := [3]float32{res[0][0], res[1][0], res[2][0]}
	expect := [3]float32{-0.03450077, 0.21015842, 0.9770585}
	fmt.Println("result:", got)
	if got != expect {
		fmt.Println("expected:", expect)
		os.Exit(2)
	} else {
		fmt.Println("OK")
	}
}
