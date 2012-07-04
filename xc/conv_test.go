package xc

import (
	"testing"
	"nimble-cube/core"
)

func TestConv(test *testing.T) {
	size := [3]int{2, 4, 8}
	core.InitSize(size[0], size[1], size[2])
	N := prod(size)

	in := make([]float32, 3*N)
	input := [3][]float32{in[0*N : 1*N], in[1*N : 2*N], in[2*N : 3*N]}

	out := make([]float32, 3*N)
	output := [3][]float32{out[0*N : 1*N], out[1*N : 2*N], out[2*N : 3*N]}

	conv := NewConv(input, output, size)

	conv.Push(1)
	conv.Push(2)
	conv.Push(3)
	conv.Push(4)
	conv.Push(5)
	conv.Push(core.N())
}
