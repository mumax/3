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
	SetOD("gpu4")
	// mesh

	a := 8
	N0, N1, N2 := 1, a*32, a*128
	cx, cy, cz := 3e-9, 3.125e-9, 3.125e-9
	mesh := NewMesh(N0, N1, N2, cx, cy, cz)
	size := mesh.Size()
	Log("mesh:", mesh)
	Log("block:", BlockSize(mesh.Size()))

	// quantities

	mGPU := gpu.MakeChan3("m", "", mesh)
	b := gpu.MakeChan3("B", "T", mesh)

	acc := 1
	kernel := mag.BruteKernel(mesh, acc)
	Stack(conv.NewSymm2D(size, kernel, mGPU.NewReader(), b))

	const Msat = 1.0053
	aex := mag.Mu0 * 13e-12 / Msat
	//bex := gpu.MakeChan3("Bex", "T", mesh)
	bex := gpu.RunExchange6("Bex", mGPU, aex).Output()

	//bexH := MakeChan3(size, "BexH")
	//	Stack(conv.NewDownloader(bex.MakeRChan3(), bexH))
	//	Stack(dump.NewAutosaver("BexH", bexH.MakeRChan3(), 100))

	beffGPU := gpu.MakeChan3("Beff", "T", mesh)
	Stack(gpu.NewAdder3(beffGPU, b.NewReader(), Msat, bex.NewReader(), 1))

	var alpha float32 = 1
	//torque := gpu.MakeChan3(size, "τ")
	torque := gpu.RunLLGTorque("torque", mGPU, beffGPU, alpha).Output()

	dt := 50e-15
	solver := gpu.NewEuler(mGPU, torque.NewReader(), dt, mag.Gamma0)

	mHost := MakeChan3("mHost", "", mesh)
	Stack(conv.NewDownloader(mGPU.NewReader(), mHost))
	Stack(dump.NewAutosaver("m.dump", mHost.NewReader(), 100))

	Stack(dump.NewAutosaver("test4m.dump", mHost.NewReader(), 100))
	Stack(dump.NewAutotable("test4m.table", mHost.NewReader(), 100))

	RunStack()

	in := MakeVectors(size)
	mag.SetAll(in, mag.Uniform(0, 0.1, 1))
	for i := 0; i < 3; i++ {
		mGPU.UnsafeData()[i].CopyHtoD(Contiguous(in[i]))
	}

	gpu.LockCudaThread()
	solver.Steps(20)

	ProfDump(os.Stdout)
	Cleanup()
}
