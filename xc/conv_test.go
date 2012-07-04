package xc

import (
	"testing"
)

func TestConv(test *testing.T) {
	size := [3]int{2, 4, 8}
	N := prod(size)

	in := make([]float32, 3*N)
	input := [3][]float32{in[0*N : 1*N], in[1*N : 2*N], in[2*N : 3*N]}

	out := make([]float32, 3*N)
	output := [3][]float32{out[0*N : 1*N], out[1*N : 2*N], out[2*N : 3*N]}

	var conv Conv
	conv.Init(input, output, size)
}
