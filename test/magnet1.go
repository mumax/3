package main

import (
	"flag"
	"math"
	"nimble-cube/core"
	"nimble-cube/dump"
	"nimble-cube/gpu/conv"
	"nimble-cube/mag"
	"strconv"
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
	m := demag.Input()
	m_ := core.Contiguous3(m)
	Hd := demag.Output()
	Hd_:= core.Contiguous3(Hd)

	// inital mag
	Θ := 5.
	mag.Uniform(m, mag.Vector{0, float32(math.Sin(Θ)), float32(math.Cos(Θ))})
	dump.Quick("m", m[:])

	demag.Exec()
	dump.Quick("hd", Hd[:])

	Hex := core.MakeVectors(size)
	Hex_:=core.Contiguous3(Hex)
	mag.Exchange6(m, Hex, cellsize)	
	dump.Quick("hex", Hex[:])

	Heff := core.MakeVectors(size)
	Heff_ := core.Contiguous3(Heff)
	core.Add3(Heff_, Hex_, Hd_)
	dump.Quick("heff", Heff[:])

	alpha:=float32(1)
	torque := core.MakeVectors(size)
	torque_ := core.Contiguous3(torque)
	mag.LLGTorque(torque_, m_, Heff_, alpha)
	dump.Quick("torque", torque[:])

	N:=1000
	for step := 0; step < N; step++{
		demag.Exec()
		mag.Exchange6(m, Hex, cellsize)	
		core.Add3(Heff_, Hex_, Hd_)
		mag.LLGTorque(torque_, m_, Heff_, alpha)
	}	

}

func intflag(idx int) int {
	val, err := strconv.Atoi(flag.Arg(idx))
	core.Fatal(err)
	return val
}
