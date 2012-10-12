package main

import (
	. "nimble-cube/core"
	"nimble-cube/dump"
	"nimble-cube/gpu/conv"
	"nimble-cube/mag"
	"nimble-cube/gpu"
	"os"
)

func main() {

	// mesh

	a := 1
	N0, N1, N2 := 1, a*32, a*128
	cx, cy, cz := 3e-9, 3.125e-9, 3.125e-9
	mesh := NewMesh(N0, N1, N2, cx, cy, cz)
	size := mesh.GridSize()
	Log("mesh:", mesh)
	Log("block:", BlockSize(mesh.GridSize()))

	// quantities

	m := MakeChan3(size, "m")
	mGPU := gpu.MakeChan3(size, "mGPU")
	Stack(conv.NewUploader(m.MakeRChan3() , mGPU))

	b := gpu.MakeChan3(size, "B")

	acc := 1
	kernel := mag.BruteKernel(mesh.ZeroPadded(), acc)
	Stack(conv.NewSymm2D(size, kernel, mGPU.MakeRChan3(), b))

	Msat := 1.0053
	aex := Mu0 * 13e-12 / Msat
	bex := gpu.MakeChan3(size, "Bex")
	Stack(gpu.NewExchange6(mGPU.MakeRChan3(), bex, mesh, aex))

	beGPU := gpu.MakeChan3(size, "BeGPU")
	Stack(gpu.NewAdder3(beGPU, bex.MakeRChan3(), b.MakeRChan3()))

	be := MakeChan3(size, "Be")
	Stack(conv.NewDownloader(beGPU.MakeRChan3(), be))
	

	var alpha float32 = 1
	torque := MakeChan3(size, "τ")
	Stack(mag.NewLLGTorque(torque, m.MakeRChan3(), be.MakeRChan3(), alpha))

	var dt float32 = 100e-15
	solver := mag.NewEuler(m, torque.MakeRChan3(), dt)
	mag.SetAll(m.UnsafeArray(), mag.Uniform(0, 0.1, 1))
	Stack(dump.NewAutosaver("test4m.dump", m.MakeRChan3(), 100))
	//Stack(dump.NewAutosaver("test4bex.dump", bex.MakeRChan3(), 100))
	Stack(dump.NewAutotable("test4m.table", m.MakeRChan3(), 100))

	RunStack()

	solver.Steps(100)

	ProfDump(os.Stdout)
	Cleanup()
}
