package main

import (
	. "nimble-cube/core"
	"nimble-cube/dump"
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
	hd := MakeChan3(size, "hd")

	acc := 8
	kernel := mag.BruteKernel(mesh.ZeroPadded(), acc)
	Stack(conv.NewSymmetricHtoD(size, kernel, m.MakeRChan3(), hd))

	Msat := 1.0053
	aex := Mu0 * 13e-12 / Msat
	hex := MakeChan3(size, "hex")
	Stack(mag.NewExchange6(m.MakeRChan3(), hex, mesh, aex))

	heff := MakeChan3(size, "heff")
	Stack(NewAdder3(heff, hd.MakeRChan3(), hex.MakeRChan3()))

	const alpha = 1
	torque := MakeChan3(size, "torque")
	Stack(mag.NewLLGTorque(torque, m.MakeRChan3(), heff.MakeRChan3(), alpha))

	const dt = 100e-15
	solver := mag.NewEuler(m, torque.MakeRChan3(), dt)
	mag.SetAll(m.UnsafeArray(), mag.Uniform(0, 0.1, 1))
	//Stack(dump.NewAutosaver("m.dump", m.MakeRChan3(), 100))
	Stack(dump.NewAutotable("m.table", m.MakeRChan3(), 10))

	RunStack()

	solver.Steps(100)
	// TODO: drain

	ProfDump(os.Stdout)
	Cleanup()
}
