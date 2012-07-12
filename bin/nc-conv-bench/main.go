package main

/*
	This program tests and benchmarks the micormagnetic convolution
*/

import (
	"flag"
	"fmt"
	"nimble-cube/core"
	. "nimble-cube/xc"
	"strconv"
	"time"
)

func main() {
	n0, _ := strconv.Atoi(flag.Arg(0))
	n1, _ := strconv.Atoi(flag.Arg(1))
	n2, _ := strconv.Atoi(flag.Arg(2))

	size := [3]int{n0, n1, n2}
	core.InitSize(size[0], size[1], size[2])
	N := size[0] * size[1] * size[2]

	in := make([]float32, 3*N)
	input := [3][]float32{in[0*N : 1*N], in[1*N : 2*N], in[2*N : 3*N]}

	out := make([]float32, 3*N)
	output := [3][]float32{out[0*N : 1*N], out[1*N : 2*N], out[2*N : 3*N]}

	conv := NewConv(input, output, size)

	conv.Test()
	conv.Push(core.N())
	conv.Pull(core.N())

	loops := [...]int{30}
	for _, loop := range loops {
		start := time.Now()
		fmt.Println("running", loop, "loops")
		go func() {
			for i := 0; i < loop; i++ {
				conv.Push(core.N())
			}
		}()
		for i := 0; i < loop; i++ {
			conv.Pull(core.N())
		}
		duration := time.Since(start) / time.Duration(loop)
		fmt.Println(duration, "/op")
		bytes := 4 * 3 * 2 * N
		seconds := float64(duration) / float64(time.Second)
		fmt.Println("bandwidth:", float32((float64(bytes)/seconds)/1000000), "MB/s")
	}
	core.CleanExit()
}
