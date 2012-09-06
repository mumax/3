package main

import (
	"fmt"
	"nimble-cube/core"
	"nimble-cube/dump"
	"nimble-cube/gpu/conv"
	"nimble-cube/mag"
	"os"
)

func main() {
	N0, N1, N2 := 1, 128, 128
	size := [3]int{N0, N1, N2}
	cellsize := [3]float64{1e-9, 1e-9, 1e-9}

	acc := 8
	noPBC := [3]int{0, 0, 0}
	kernel := mag.BruteKernel(core.PadSize(size, noPBC), cellsize, noPBC, acc)
	demag := conv.NewSymmetric(size, kernel)

	m := demag.Input()
	h := demag.Output()
	mag.Uniform(m, mag.Vector{1, 0, 0})
	demag.Exec()

	dump.Quick("h_demag", h[:])

	h0, h1, h2 := h[0][0][N1/2][N2/2], h[1][0][N1/2][N2/2], h[2][0][N1/2][N2/2]
	fmt.Println("H_demag(center):", h0, h1, h2)
	if h1 != 0 || h2 != 0 || h0 > -0.99 || h0 < -1 {
		os.Exit(1)
	}
}
