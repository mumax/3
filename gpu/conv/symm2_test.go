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
	hin := [3]core.Chan{core.MakeChan(s), core.MakeChan(s), core.MakeChan(s)}
	hinR := [3]core.RChan{hin[0].ReadOnly(), hin[1].ReadOnly(), hin[2].ReadOnly()}
	din := [3]gpu.Chan{gpu.MakeChan(s), gpu.MakeChan(s), gpu.MakeChan(s)}
	dinR := [3]gpu.RChan{din[0].ReadOnly(), din[1].ReadOnly(), din[2].ReadOnly()}
	dout := [3]gpu.Chan{gpu.MakeChan(s), gpu.MakeChan(s), gpu.MakeChan(s)}
	doutR := [3]gpu.RChan{dout[0].ReadOnly(), dout[1].ReadOnly(), dout[2].ReadOnly()}
	hout := [3]core.Chan{core.MakeChan(s), core.MakeChan(s), core.MakeChan(s)}

	acc := 2
	kern := mag.BruteKernel(mesh.ZeroPadded(), acc)

	arr := [3][][][]float32{hin[0].Array, hin[1].Array, hin[2].Array}
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
