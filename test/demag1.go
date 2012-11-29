package main

import (
	"code.google.com/p/nimble-cube/gpu"
	"code.google.com/p/nimble-cube/mag"
	"code.google.com/p/nimble-cube/nimble"
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

	N0, N1, N2 := 1, 3*64, 5*64
	cx, cy, cz := 3e-9, 3.125e-9, 3.125e-9
	mesh := nimble.NewMesh(N0, N1, N2, cx, cy, cz)
	fmt.Println("mesh:", mesh)

	mbox := gpu.NewConst("m", "", mesh, nimble.UnifiedMemory, []float64{1, 0, 0})
	m := mbox.Output()
	nimble.Stack(mbox)

	acc := 2
	kernel := mag.BruteKernel(mesh, acc)
	conv := gpu.NewConvolution("B", "T", mesh, nimble.UnifiedMemory, kernel, m)
	B := conv.Output()

	const probe = 24 * 121
	outputc := B.NewReader()
	nimble.RunStack()
	output := host(outputc.ReadNext(mesh.NCell()))
	if output[0][probe] != -0.9709071 || output[1][probe] != 0 || output[2][probe] != 0 {
		fmt.Println("failed, got:", output[0][probe])
		os.Exit(2)
	} else {
		fmt.Println("OK")
	}
}
