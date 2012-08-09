package main

import (
	"nimble-cube/core"
	"nimble-cube/gpu/conv"
	"nimble-cube/mag"
)

func main() {
	size := [3]int{1, 32, 32}
	N := core.Prod(size)
	cellsize := [3]float64{1e-9, 1e-9, 1e-9}

	input := core.MakeVectors(size)
	output := core.MakeVectors(size)

	pbc := [3]int{0, 0, 0}
	acc := 4
	kern := mag.BruteKernel(size, cellsize, pbc, acc)
	c := conv.NewSymmetric(input, output, kern)
	c.Push(N)
	c.Pull(N)

}
