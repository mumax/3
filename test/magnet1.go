package main

import (
	"math"
	"strconv"
	"flag"
	"nimble-cube/core"
	"nimble-cube/dump"
	"nimble-cube/gpu/conv"
   	"nimble-cube/mag"
)

func main() {
		flag.Parse()
		N0, N1, N2 := intflag(0), intflag(1), intflag(2)
		//N := N0 * N1 * N2
		size := [3]int{N0, N1, N2}
		cellsize := [3]float64{3e-9, 3.125e-9, 3.125e-9}
	
		// demag
		acc := 4
		kernel := mag.BruteKernel(size, cellsize, [3]int{0, 0, 0}, acc)
		demag := conv.NewSymmetric(size, kernel)
		m:=demag.Input()
		Hd := demag.Output()

		// inital mag
		Θ := 0.
		mag.Uniform(m, mag.Vector{0, float32(math.Sin(Θ)), float32(math.Cos(Θ))})
		dump.Quick("m", m[:])

		demag.Exec()
		dump.Quick("hd", Hd[:])

	
}

func intflag(idx int)int{
		val, err := strconv.Atoi(flag.Arg(idx))
	core.Fatal(err)		
	return val
}
