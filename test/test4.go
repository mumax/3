package main

import (
	. "nimble-cube/core"
	"nimble-cube/gpu/conv"
	"nimble-cube/mag"
)

func main() {
	N0, N1, N2 := 1, 32, 128
	cx, cy, cz := 3e-9, 3.125e-9, 3.125e-9
	mesh := NewMesh(N0, N1, N2, cx, cy, cz)
	size := mesh.GridSize()

	m1 := MakeChan3(size)
	heff := MakeChan3(size)

	acc := 8
	kernel := mag.BruteKernel(mesh.ZeroPadded(), acc)
	go conv.NewSymmetricHtoD(size, kernel, m1.MakeRChan3(), heff).Run()

}
