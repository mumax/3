package conv

import (
	"code.google.com/p/nimble-cube/gpu"
	"code.google.com/p/nimble-cube/mag"
	"code.google.com/p/nimble-cube/nimble"
)

// Internal: main function for conv test.
func TestSymm2(N0, N1, N2 int) {
	C := 1e-9
	mesh := nimble.NewMesh(N0, N1, N2, C, 2*C, 3*C)
	nimble.Log(mesh)
	N := mesh.NCell()

	gpu.LockCudaThread()
	hin := nimble.MakeChan3("hin", "", mesh, nimble.UnifiedMemory)
	hout := nimble.MakeChan3("hout", "", mesh, nimble.UnifiedMemory)

	acc := 1
	kern := mag.BruteKernel(nimble.ZeroPad(mesh), acc)

	arr := hin.UnsafeArray()
	initConvTestInput(arr)

	F := 10
	go func() {
		for i := 0; i < F; i++ {
			hin.WriteNext(N)
			hin.WriteDone()
		}
	}()

	//go NewSymmetricHtoD(mesh, kern, hin.NewReader(), hout).Run()
	go NewSymm2D(mesh, kern, hin, hout).Run()

	houtR := hout.NewReader()
	for i := 0; i < F; i++ {
		houtR.ReadNext(N)
		houtR.ReadDone()
	}

	outarr := hout.UnsafeData()

	ref := nimble.MakeVectors(mesh.Size())
	Brute(arr, ref, kern)
	checkErr(outarr, nimble.Contiguous3(ref))
}
