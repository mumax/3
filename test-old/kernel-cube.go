package main

import (
	"code.google.com/p/mx3/core"
	"code.google.com/p/mx3/mag"
	"code.google.com/p/mx3/nimble"
	"fmt"
)

// Precision test for the kernel: cube.
func main() {
	nimble.Init()
	core.LOG = false
	defer nimble.Cleanup()
	nimble.SetOD("kernel-cube.out")

	N0, N1, N2 := 2, 2, 2
	cx, cy, cz := 1e-9, 1e-9, 1e-9
	mesh := nimble.NewMesh(N0, N1, N2, cx, cy, cz)
	fmt.Println("mesh:", mesh)

	for acc := 1; acc < 10; acc++ {
		mag.BruteKernel(mesh, float64(acc))
		//fmt.Println(acc, kernel[0][0][0][0][0], kernel[1][0][0][0][0])
	}
}
