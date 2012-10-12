package main

import (
	. "nimble-cube/core"
	"nimble-cube/dump"
	"nimble-cube/gpu"
	"nimble-cube/gpu/conv"
	"nimble-cube/mag"
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

	mGPU := gpu.MakeChan3(size, "mGPU")
	b := gpu.MakeChan3(size, "B")

	acc := 1
	kernel := mag.BruteKernel(mesh.ZeroPadded(), acc)
	Stack(conv.NewSymm2D(size, kernel, mGPU.MakeRChan3(), b))

	const Msat = 1.0053
	aex := mag.Mu0 * 13e-12 / Msat
	bex := gpu.MakeChan3(size, "Bex")
	Stack(gpu.NewExchange6(mGPU.MakeRChan3(), bex, mesh, aex))

	bexH := MakeChan3(size, "BexH")
	Stack(conv.NewDownloader(bex.MakeRChan3(), bexH))
	Stack(dump.NewAutosaver("BexH", bexH.MakeRChan3(), 100))

	beffGPU := gpu.MakeChan3(size, "Beff")
	Stack(gpu.NewAdder3(beffGPU, b.MakeRChan3(), Msat, bex.MakeRChan3(), 1))

	var alpha float32 = 1
	torque := gpu.MakeChan3(size, "τ")
	Stack(gpu.NewLLGTorque(torque, mGPU.MakeRChan3(), beffGPU.MakeRChan3(), alpha))

	dt := 50e-15
	solver := gpu.NewEuler(mGPU, torque.MakeRChan3(), dt, mag.Gamma)

	//	mHost := MakeChan3(size, "mHost")
	//	Stack(conv.NewDownloader(mGPU.MakeRChan3(), mHost))
	//	Stack(dump.NewAutosaver("m.dump", mHost.MakeRChan3(), 100))

	//Stack(dump.NewAutosaver("test4m.dump", m.MakeRChan3(), 100))
	//Stack(dump.NewAutotable("test4m.table", m.MakeRChan3(), 100))
	//Stack(dump.NewAutosaver("test4bex.dump", bex.MakeRChan3(), 100))

	RunStack()

	in := MakeVectors(size)
	mag.SetAll(in, mag.Uniform(0, 0.1, 1))
	for i := 0; i < 3; i++ {
		mGPU.UnsafeData()[i].CopyHtoD(Contiguous(in[i]))
	}

	gpu.LockCudaThread()
	solver.Steps(20000)

	ProfDump(os.Stdout)
	Cleanup()
}
