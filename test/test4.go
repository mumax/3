package main

import (
	"code.google.com/p/nimble-cube/gpu"
	"code.google.com/p/nimble-cube/gpu/conv"
	"code.google.com/p/nimble-cube/mag"
	"code.google.com/p/nimble-cube/nimble"
	"fmt"
)

func main() {
	nimble.Init()
	defer nimble.Cleanup()
	nimble.SetOD("test4.out")

	N0, N1, N2 := 1, 32, 128
	cx, cy, cz := 3e-9, 3.125e-9, 3.125e-9
	mesh := nimble.NewMesh(N0, N1, N2, cx, cy, cz)
	fmt.Println("mesh:", mesh)

	m := gpu.NewConst("m", "", mesh, 1, 0, 0).Output().Chan3()
	fmt.Println("m:", m)

	acc := 8
	kernel := mag.BruteKernel(mesh, acc)
	B := nimble.MakeChan3("B", "T", mesh, nimble.GPUMemory, -1)
	demag := conv.NewSymm2D(mesh, kernel, m, B)
	nimble.Stack(demag)

	//	Msat := 1.0053
	//	aex := mag.Mu0 * 13e-12 / Msat
	//	hex := MakeChan3("Hex", "", mesh)
	//	Stack(mag.NewExchange6(m.NewReader(), hex, mesh, aex))
	//
	//	heff := MakeChan3("Heff", "", mesh)
	//	Stack(NewAdder3(heff, hd.NewReader(), hex.NewReader()))
	//
	//	const alpha = 1
	//	torque := MakeChan3("τ", "", mesh)
	//	Stack(mag.NewLLGTorque(torque, m.NewReader(), heff.NewReader(), alpha))
	//
	//	const dt = 50e-15
	//	solver := mag.NewEuler(m, torque.NewReader(), mag.Gamma0, dt)
	//	mag.SetAll(m.UnsafeArray(), mag.Uniform(0, 0.1, 1))
	//	Stack(dump.NewAutosaver("h.dump", hd.NewReader(), 1))
	//	Stack(dump.NewAutosaver("m.dump", m.NewReader(), 1))
	//	Stack(dump.NewAutosaver("hex.dump", hex.NewReader(), 1))
	//	Stack(dump.NewAutotable("m.table", m.NewReader(), 1))
	//
	//	RunStack()
	//
	//	solver.Steps(100)
	//	res := m.UnsafeArray()
	//	got := [3]float32{res[0][0][0][0], res[1][0][0][0], res[2][0][0][0]}
	//	expect := [3]float32{-0.075877085, 0.17907967, 0.9809043}
	//	Log("result:", got)
	//	if got != expect {
	//		Fatal(fmt.Errorf("expected: %v", expect))
	//	}
	//solver.Steps(10000)

}
