package main

import (
	"nimble-cube/dump"
	"nimble-cube/gpu/conv"
	"nimble-cube/nimble"
	//"nimble-cube/mag"
	"flag"
	"strconv"
)

func main() {
	flag.Parse()
	N0, _ := strconv.Atoi(flag.Arg(0))
	N1, _ := strconv.Atoi(flag.Arg(1))
	N2, _ := strconv.Atoi(flag.Arg(2))

	size := [3]int{N0, N1, N2}
	nimble.Log("size:", size)

	ksize := nimble.PadSize(size, [3]int{0, 0, 0})
	//kern := mag.BruteKernel(ksize, [3]float64{1, 2, 3}, [3]int{0, 0, 0}, acc)

	var kern [3][3][][][]float32
	for i := 0; i < 3; i++ {
		for j := 0; j < 3; j++ {
			kern[i][j] = nimble.MakeFloats(ksize)
		}
	}
	kern[2][0][0][0][0] = 1

	dump.Quick("kx.dump", kern[0][:])
	dump.Quick("ky.dump", kern[1][:])
	dump.Quick("kz.dump", kern[2][:])

	c := conv.NewGeneral(size, kern)
	input := c.Input()
	output := c.Output()

	for i := range input[0] {
		for j := range input[0][0] {
			for k := range input[0][0][0] {
				input[0][i][j][k] = float32(k)
			}
		}
	}

	//	input[0][N0-1][N1-1][N2-1] = 1
	//	input[1][0][0][0] = 0
	//	input[2][0][0][0] = 0
	//input[2][N0-1][N1-1][N2-1] = 1

	conv.Brute(input, output, kern)
	dump.Quick("brute.dump", output[:])

	c.Exec()

	dump.Quick("input.dump", input[:])
	dump.Quick("output.dump", output[:])

}
