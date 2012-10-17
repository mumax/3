package main

import (
	"fmt"
	. "nimble-cube/core"
	"nimble-cube/gpu/conv"
	"nimble-cube/mag"
	"os"
)

func main() {
	N0, N1, N2 := 1, 32, 128
	cx, cy, cz := 3e-9, 3.125e-9, 3.125e-9
	mesh := NewMesh(N0, N1, N2, cx, cy, cz)
	size := mesh.GridSize()
	Log("mesh:", mesh)
	Log("block:", BlockSize(mesh.GridSize()))

	m := MakeChan3(size, "m")
	hd := MakeChan3(size, "Hd")

	acc := 8
	kernel := mag.BruteKernel(mesh.ZeroPadded(), acc)
	Stack(conv.NewSymmetricHtoD(size, kernel, m.MakeRChan3(), hd))

	Msat := 1.0053
	aex := mag.Mu0 * 13e-12 / Msat
	hex := MakeChan3(size, "Hex")
	Stack(mag.NewExchange6(m.MakeRChan3(), hex, mesh, aex))

	heff := MakeChan3(size, "Heff")
	Stack(NewAdder3(heff, hd.MakeRChan3(), hex.MakeRChan3()))

	const alpha = 0.02
	torque := MakeChan3(size, "τ")
	Stack(mag.NewLLGTorque(torque, m.MakeRChan3(), heff.MakeRChan3(), alpha))

	const dt = 100e-15
	solver := mag.NewEuler(m, torque.MakeRChan3(), mag.Gamma, dt)
	mag.SetAll(m.UnsafeArray(), mag.Uniform(0, 0.1, 1))
	//	Stack(dump.NewAutosaver("ex2dm.dump", m.MakeRChan3(), 100))
	//	Stack(dump.NewAutosaver("ex2dhex.dump", hex.MakeRChan3(), 100))
	//	Stack(dump.NewAutotable("ex2dm.table", m.MakeRChan3(), 10))

	RunStack()

	solver.Steps(100)
	res := m.UnsafeArray()
	got := [3]float32{res[0][0][0][0], res[1][0][0][0], res[2][0][0][0]}
	expect := [3]float32{-0.075877085, 0.17907967, 0.9809043}
	Log("result:", got)
	if got != expect {
		Fatal(fmt.Errorf("expected: %v", expect))
	}
	// TODO: drain

	ProfDump(os.Stdout)
	Cleanup()
}
