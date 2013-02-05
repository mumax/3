package main

import (
	//"code.google.com/p/mx3/dump"
	"code.google.com/p/mx3/gpu/conv"
	"code.google.com/p/mx3/mag"
	"code.google.com/p/mx3/nimble"
	"fmt"
	"os"
)

func main() {
	nimble.Init()
	nimble.SetOD("demag.out")

	N0, N1, N2 := 1, 32, 128
	cx, cy, cz := 3e-9, 3.125e-9, 3.125e-9
	mesh := nimble.NewMesh(N0, N1, N2, cx, cy, cz)
	fmt.Println("mesh:", mesh)

	//testM :=
	m := nimble.NewConstant("m", "", mesh, testM).Output()

	const acc = 10
	kernel := mag.BruteKernel(ZeroPad(mesh), acc)
	demag := conv.NewSymm2D(mesh, kernel, m)
	hd := demag.Output()

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

	ProfDump(os.Stdout)
	Cleanup()
}
