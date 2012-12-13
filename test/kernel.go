package main

import (
	"code.google.com/p/mx3/gpu"
	"code.google.com/p/mx3/mag"
	"code.google.com/p/mx3/core"
	"code.google.com/p/mx3/nimble"
	"fmt"
	"os"
	"math"
)

// Precision test for the kernel
func main() {
	nimble.Init()
	defer nimble.Cleanup()
	nimble.SetOD("kernel.out")

	Y, X := core.IntArg(0), core.IntArg(1)
	y, x := float64(Y), float64(X)
	N0, N1, N2 := 1, 1024/Y, 1024/X
	cx, cy, cz := 1e-9, 1e-9*y, 1e-9*x
	mesh := nimble.NewMesh(N0, N1, N2, cx, cy, cz)
	fmt.Println("mesh:", mesh)

	mbox := gpu.NewConst("m", "", mesh, nimble.UnifiedMemory, []float64{1, 0, 0})
	m := mbox.Output()

	acc := 4.
	kernel := mag.BruteKernel(mesh, acc)
	conv := gpu.NewConvolution("B", "T", mesh, nimble.UnifiedMemory, kernel, m)
	B := conv.Output()

	outputc := B.NewReader()
	nimble.RunStack()
	output := host(outputc.ReadNext(mesh.NCell()))
	Bz := core.Reshape(output[0], [3]int{N0, N1, N2})
	probe := float64(Bz[N0/2][N1/2][N2/2])
	fmt.Println(probe)
	want := -1.
	if math.Abs(probe-want) > 0.01 {
		fmt.Println("FAIL")
		os.Exit(2)
	}else{
		fmt.Println("OK")
	}
}


func host(s []nimble.Slice) [][]float32 {
	h := make([][]float32, len(s))
	for i := range h {
		h[i] = s[i].Host()
	}
	return h
}

