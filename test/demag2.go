package main

import (
	"code.google.com/p/mx3/core"
	"code.google.com/p/mx3/gpu"
	"code.google.com/p/mx3/mag"
	"code.google.com/p/mx3/nimble"
	"fmt"
	"os"
)

func host(s []nimble.Slice) [][]float32 {
	h := make([][]float32, len(s))
	for i := range h {
		h[i] = s[i].Host()
	}
	return h
}

func main() {
	nimble.Init()
	defer nimble.Cleanup()
	nimble.SetOD("demag1.out")

	N0, N1, N2 := 4, 32, 1024
	cx, cy, cz := 3e-9, 3.125e-9, 3.125e-9
	mesh := nimble.NewMesh(N0, N1, N2, cx, cy, cz)
	fmt.Println("mesh:", mesh)

	mbox := gpu.NewConst("m", "", mesh, nimble.UnifiedMemory, []float64{1, 0, 0})
	m := mbox.Output()

	acc := 3
	kernel := mag.BruteKernel(mesh, acc)
	conv := gpu.NewConvolution("B", "T", mesh, nimble.UnifiedMemory, kernel, m)
	B := conv.Output()

	outputc := B.NewReader()
	nimble.RunStack()
	output := host(outputc.ReadNext(mesh.NCell()))

	out0 := core.Reshape(output[0], mesh.Size())
	out1 := core.Reshape(output[1], mesh.Size())
	out2 := core.Reshape(output[2], mesh.Size())
	X, Y, Z := N0/2, N1/2, N2/2
	if out0[X][Y][Z] != -0.9239292 || out1[X][Y][Z] > 0.001 || out2[X][Y][Z] > 0.001 {
		fmt.Println("failed, got:", out0[X][Y][Z], out1[X][Y][Z], out2[X][Y][Z])
		os.Exit(2)
	} else {
		fmt.Println("OK")
	}
}
