package main

import (
	"code.google.com/p/nimble-cube/cpu"
	//"code.google.com/p/nimble-cube/gpu"
	"code.google.com/p/nimble-cube/gpu/conv"
	"code.google.com/p/nimble-cube/mag"
	"code.google.com/p/nimble-cube/nimble"
	"fmt"
)

func main() {
	nimble.Init()
	defer nimble.Cleanup()
	nimble.SetOD("test4.out")

	N0, N1, N2 := 1, 32, 64
	cx, cy, cz := 3e-9, 3.125e-9, 3.125e-9
	mesh := nimble.NewMesh(N0, N1, N2, cx, cy, cz)
	fmt.Println("mesh:", mesh)

	m := nimble.MakeChan3("m", "", mesh, nimble.UnifiedMemory, 0)
	fmt.Println("m:", m)

	acc := 8
	kernel := mag.BruteKernel(mesh, acc)
	B := conv.NewSymm2D("B", "T", mesh, nimble.UnifiedMemory, kernel, m).Output()

	const Bsat = 1.0053
	const aex = mag.Mu0 * 13e-12 / Bsat
	exch := cpu.NewExchange6("Bex", "T", nimble.UnifiedMemory, m.NewReader(), aex)
	Bex := exch.Output()

	//	heff := MakeChan3("Heff", "", mesh)
	Beff := cpu.NewSum("Beff", B, Bex, Bsat, 1, nimble.UnifiedMemory).Output().Chan3()

	const alpha = 1
	torque := nimble.MakeChan3("τ", "", mesh, nimble.UnifiedMemory, 1)
	nimble.Stack(cpu.NewLLGTorque(torque, m.NewReader(), Beff.NewReader(), alpha))

	const dt = 50e-15

	solver := cpu.NewEuler(m, torque.NewReader(), mag.Gamma0, dt)

	M := cpu.Host(m.ChanN().UnsafeData())
	for i := range M[2] {
		M[2][i] = 1
	}

	//	Stack(dump.NewAutosaver("h.dump", hd.NewReader(), 1))
	//	Stack(dump.NewAutosaver("m.dump", m.NewReader(), 1))
	//	Stack(dump.NewAutosaver("hex.dump", hex.NewReader(), 1))
	//	Stack(dump.NewAutotable("m.table", m.NewReader(), 1))

	nimble.RunStack()

	solver.Steps(100)
	//	res := m.UnsafeArray()
	//	got := [3]float32{res[0][0][0][0], res[1][0][0][0], res[2][0][0][0]}
	//	expect := [3]float32{-0.075877085, 0.17907967, 0.9809043}
	//	Log("result:", got)
	//	if got != expect {
	//		Fatal(fmt.Errorf("expected: %v", expect))
	//	}
	//solver.Steps(10000)

}
