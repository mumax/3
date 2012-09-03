package main

import(
	"nimble-cube/core"
	"nimble-cube/dump"
	"nimble-cube/gpu/conv"
	"nimble-cube/mag"
	"strconv"
	"flag"
)


func main(){
	flag.Parse()
	N0, _ := strconv.Atoi(flag.Arg(0))
	N1 , _:= strconv.Atoi(flag.Arg(1))
	N2, _ := strconv.Atoi(flag.Arg(2))

	size := [3]int{N0, N1, N2}
	core.Log("size:", size)

	input := core.MakeVectors(size)
	input[0][0][0][0] = 1
	input[1][0][0][0] = 0
	input[2][0][0][0] = 0
	dump.Quick("input.dump", input[:])

	output := core.MakeVectors(size)

	ksize := core.PadSize(size, [3]int{0, 0, 0})
	acc := 4
	kern := mag.BruteKernel(ksize, [3]float64{1, 2, 3}, [3]int{0, 0, 0}, acc)
	dump.Quick("kx.dump", kern[0][:])
	dump.Quick("ky.dump", kern[1][:])
	dump.Quick("kz.dump", kern[2][:])

	c :=conv.NewGeneral(input, output, kern)
	c.Exec()
	dump.Quick("output.dump", output[:])

	conv.Brute(input, output, kern)
	dump.Quick("brute.dump", output[:])
}
