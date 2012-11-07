package main

import (
	"nimble-cube/dump"
	"nimble-cube/gpu"
	"nimble-cube/gpu/conv"
	"nimble-cube/mag"
	. "nimble-cube/nimble"
	"os"
)

func main() {
	SetOD("gpu4.out")
	// mesh

	N0, N1, N2 := 1, 32, 128
	cx, cy, cz := 3e-9, 3.125e-9, 3.125e-9
	mesh := NewMesh(N0, N1, N2, cx, cy, cz)
	size := mesh.Size()
	Log("mesh:", mesh)
	Log("block:", BlockSize(mesh.Size()))

	// quantities

	mGPU := gpu.MakeChan3("m", "", mesh)
	b := gpu.MakeChan3("B", "T", mesh)

	acc := 8
	kernel := mag.BruteKernel(mesh, acc)
	Stack(conv.NewSymm2D(size, kernel, mGPU.NewReader(), b))

	const Msat = 1.0053
	aex := mag.Mu0 * 13e-12 / Msat
	//bex := gpu.MakeChan3("Bex", "T", mesh)
	bex := gpu.RunExchange6("Bex", mGPU, aex).Output()

	beffGPU := gpu.MakeChan3("Beff", "T", mesh)
	Stack(gpu.NewAdder3(beffGPU, b.NewReader(), Msat, bex.NewReader(), 1))

	var alpha float32 = 1
	//torque := gpu.MakeChan3(size, "τ")
	torque := gpu.RunLLGTorque("torque", mGPU, beffGPU, alpha).Output()

	dt := 100e-15
	//solver := gpu.NewHeun(mGPU, torque, dt, mag.Gamma0)
	solver := gpu.NewEuler(mGPU, torque.NewReader(), dt, mag.Gamma0)

	mHost := MakeChan3("mHost", "", mesh)
	Stack(conv.NewDownloader(mGPU.NewReader(), mHost))
	Stack(dump.NewAutosaver("m.dump", mHost.NewReader(), 1))
	Stack(dump.NewAutotable("m.table", mHost.NewReader(), 1))
	bH := MakeChan3("B", "T", mesh)
	Stack(conv.NewDownloader(b.NewReader(), bH))
	Stack(dump.NewAutosaver("B.dump", bH.NewReader(), 100))

	in := MakeVectors(size)
	mag.SetAll(in, mag.Uniform(0, 0.1, 1))
	for i := 0; i < 3; i++ {
		mGPU.UnsafeData()[i].CopyHtoD(Contiguous(in[i]))
	}

	RunStack()

	gpu.LockCudaThread()
	solver.Steps(100)

	ProfDump(os.Stdout)
	Cleanup()
}
