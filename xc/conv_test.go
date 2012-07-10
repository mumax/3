package xc

import (
	"nimble-cube/core"
	"testing"
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

	input[0][0] = 1
	for i := range input[1] {
		input[1][i] = 1
	}
	for i := range input[2] {
		input[2][i] = float32(i)
	}

	conv.Push(1)
	conv.Push(6)
	conv.Push(core.N())
	conv.Push(1)
	conv.Push(core.N())
	conv.Push(1)
	conv.Push(core.N())
}
