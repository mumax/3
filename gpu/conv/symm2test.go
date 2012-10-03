package conv

import (
	"nimble-cube/core"
	"nimble-cube/gpu"
	"nimble-cube/mag"
)

// Internal: main function for conv test.
func TestSymm2(N0, N1, N2 int) {
	C := 1e-9
	mesh := core.NewMesh(N0, N1, N2, C, 2*C, 3*C)
	core.Log(mesh)
	N := mesh.NCell()
	s := mesh.GridSize()

	gpu.LockCudaThread()
	hin := core.MakeChan3(s, "hin")
	hout := core.MakeChan3(s, "hout")

	acc := 1
	kern := mag.BruteKernel(mesh.ZeroPadded(), acc)

	arr := hin.UnsafeArray()
	initConvTestInput(arr)

	F := 10
	go func() {
		for i := 0; i < F; i++ {
			hin.WriteNext(N)
			hin.WriteDone()
		}
	}()

	go NewSymmetricHtoD(mesh.GridSize(), kern, hin.MakeRChan3(), hout).Run()

	houtR := hout.MakeRChan3()
	for i := 0; i < F; i++ {
		houtR.ReadNext(N)
		houtR.ReadDone()
	}

	outarr := hout.UnsafeData()

	ref := core.MakeVectors(mesh.GridSize())
	Brute(arr, ref, kern)
	checkErr(outarr, core.Contiguous3(ref))
}
