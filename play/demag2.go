package main

import (
	"fmt"
	"nimble-cube/dump"
	"nimble-cube/gpu/conv"
	"nimble-cube/mag"
	"nimble-cube/nimble"
)

func main() {
	N0, N1, N2 := 1, 16, 32
	size := [3]int{N0, N1, N2}
	cellsize := [3]float64{1e-9, 1e-9, 1e-9}

	acc := 8
	noPBC := [3]int{0, 0, 0}
	kernel := mag.BruteKernel(nimble.PadSize(size, noPBC), cellsize, noPBC, acc)
	demag := conv.NewSymmetric(size, kernel)

	m := demag.Input()
	y1, y2 := 3*N1/8, 5*N1/8
	z1, z2 := 0*N2/8, 2*N2/8
	mag.SetRegion(m, 0, y1, z1, 1, y2, z2, mag.Uniform(0, 1, 0))

	h := demag.Output()
	demag.Exec()

	dump.Quick("h_demag", h[:])

	h0, h1, h2 := h[0][0][N1/2][N2/2], h[1][0][N1/2][N2/2], h[2][0][N1/2][N2/2]
	fmt.Println("H_demag(center):", h0, h1, h2)
}
