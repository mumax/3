package main

import (
	"math"
	"nimble-cube/core"
	"nimble-cube/gpu/conv"
	"nimble-cube/mag"
)

func main() {
	N0 := 1
	N1 := 8
	N2 := 16
	N := N0 * N1 * N2
	size := [3]int{N0, N1, N2}

	cellsize := [3]float64{3e-9, 3.125e-9, 3.125e-9}

	m := core.MakeVectors(size)
	Θ := 0.
	mag.Uniform(m, mag.Vector{0, float32(math.Sin(Θ)), float32(math.Cos(Θ))})
	core.Print("m\n", m)

	Hex := core.MakeVectors(size)
	mag.Exchange6(m, Hex, cellsize)
	core.Print("Hex\n", Hex)

	Hd := core.MakeVectors(size)
	acc := 4
	kernel := mag.BruteKernel(size, cellsize, [3]int{0, 0, 0}, acc)
	core.Print("kernel\n", kernel)
	demag := conv.NewSymmetric(m, Hd, kernel)
	demag.Push(N)
	demag.Pull(N)
	core.Print("Hd\n", Hd)

}
