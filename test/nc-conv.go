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

	ksize := core.PadSize(size, [3]int{0, 0, 0})
	acc := 4
	kern := mag.BruteKernel(ksize, [3]float64{1, 2, 3}, [3]int{0, 0, 0}, acc)
	dump.Quick("kx.dump", kern[0][:])
	dump.Quick("ky.dump", kern[1][:])
	dump.Quick("kz.dump", kern[2][:])

	c :=conv.NewGeneral(size, kern)
	input := c.Input()
	output := c.Output()
	input[0][0][0][0] = 0
	input[1][0][0][0] = 0
	input[2][N0-1][N1-1][N2-1] = 1

	conv.Brute(input, output, kern)
	dump.Quick("brute.dump", output[:])

	c.Exec()

	dump.Quick("input.dump", input[:])
	dump.Quick("output.dump", output[:])

}
