package main

import (
	"flag"
	"nimble-cube/core"
	. "nimble-cube/xc"
	"strconv"
)

func main() {
	n, _ := strconv.Atoi(flag.Arg(0))
	size := [3]int{1, n, n}
	core.InitSize(size[0], size[1], size[2])
	N := size[0] * size[1] * size[2]

	in := make([]float32, 3*N)
	input := [3][]float32{in[0*N : 1*N], in[1*N : 2*N], in[2*N : 3*N]}

	out := make([]float32, 3*N)
	output := [3][]float32{out[0*N : 1*N], out[1*N : 2*N], out[2*N : 3*N]}

	conv := NewConv(input, output, size)

	conv.Test()

	//for i := 0; i < 100; i++ {
	//	conv.Push(core.N())
	//	conv.Pull(core.N())
	//}
	core.CleanExit()
}
