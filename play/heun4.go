package main

import (
	. "nimble-cube/core"
	"nimble-cube/gpu"
	"nimble-cube/gpumag"
	"nimble-cube/mag"
)

func main() {

	// set output directory
	SetOD("heun4")

	// make mesh
	N0, N1, N2 := 1, 32, 128
	cx, cy, cz := 3e-9, 3.125e-9, 3.125e-9
	mesh := NewMesh(N0, N1, N2, cx, cy, cz)
	Log("mesh:", mesh)

	// constants

	const (
		Bsat  = 1.0053
		aex   = mag.Mu0 * 13e-12 / Bsat
		alpha = 1
		dt    = 50e-15
	)

	// add quantities
	m := gpu.MakeChan3("m", "", mesh)

	demag := gpumag.RunDemag("Bd", m) // TODO: Bsat
	Stack(demag)
	b := demag.Output()
	Log(b)

//	exch := gpu.RunExchange6("Bex", m, aex)
//	bex := exch.Output()
//	Log(bex)

	//beff := gpu.RunSum("Beff", b, Bsat, bex, 1).Output().Chan3()

	τ := gpu.RunLLGTorque("τ", m, b, alpha).Output()

	solver := gpu.NewHeun(m, τ, dt, mag.Gamma0)
	solver.Steps(1000)

	//
	//	mHost := MakeChan3(size, "mHost")
	//	Stack(conv.NewDownloader(mGPU.MakeRChan3(), mHost))
	//	Stack(dump.NewAutosaver("m.dump", mHost.MakeRChan3(), 100))
	//
	//	//Stack(dump.NewAutosaver("test4m.dump", m.MakeRChan3(), 100))
	//	//Stack(dump.NewAutotable("test4m.table", m.MakeRChan3(), 100))
	//	//Stack(dump.NewAutosaver("test4bex.dump", bex.MakeRChan3(), 100))
	//
	//	RunStack()
	//
	//	in := MakeVectors(size)
	//	mag.SetAll(in, mag.Uniform(0, 0.1, 1))
	//	for i := 0; i < 3; i++ {
	//		mGPU.UnsafeData()[i].CopyHtoD(Contiguous(in[i]))
	//	}
	//
	//	gpu.LockCudaThread()
	//
	//	ProfDump(os.Stdout)
	//	Cleanup()
}
