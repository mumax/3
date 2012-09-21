package conv

import (
	"nimble-cube/core"
	"nimble-cube/gpu"
	"nimble-cube/mag"
	"testing"
)

func TestSymm2(t *testing.T) {
	C := 1e-9
	N0, N1, N2 := 1, 4, 8
	mesh := core.NewMesh(N0, N1, N2, C, 2*C, 3*C)
	N := mesh.NCell()
	s := mesh.GridSize()

	gpu.LockCudaThread()
	hin := core.MakeChan3(s)
	hinR := hin.ReadOnly()
	din := gpu.MakeChan3(s)
	dinR := din.ReadOnly()
	dout := gpu.MakeChan3(s)
	doutR := dout.ReadOnly()
	hout := core.MakeChan3(s)

	acc := 2
	kern := mag.BruteKernel(mesh.ZeroPadded(), acc)

	arr := hin.Array()
	initConvTestInput(arr)

	go func() {
		for i := range hin {
			hin[i].WriteNext(N)
			hin[i].WriteDone()
		}
	}()

	go gpu.NewUploader(hinR[0], din[0]).Run()
	go gpu.NewUploader(hinR[1], din[1]).Run()
	go gpu.NewUploader(hinR[2], din[2]).Run()

	go NewSymm2(mesh.GridSize(), kern, dinR, dout).Run()

	go gpu.NewDownloader(doutR[0], hout[0]).Run()
	go gpu.NewDownloader(doutR[1], hout[1]).Run()
	go gpu.NewDownloader(doutR[2], hout[2]).Run()

	F := 10
	for i := 0; i < F; i++ {
		hout[0].ReadOnly().ReadNext(N)
		hout[1].ReadOnly().ReadNext(N)
		hout[2].ReadOnly().ReadNext(N)
	}

	outarr := [3][]float32{hout[0].List, hout[1].List, hout[2].List}

	ref := core.MakeVectors(mesh.GridSize())
	Brute(arr, ref, kern)
	checkErr(outarr, core.Contiguous3(ref))
}
