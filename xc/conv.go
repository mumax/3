package xc

import (
	"fmt"
	"github.com/barnex/cuda4/safe"
)

type Conv struct {
	size          [3]int
	input, output [3][]float32
	realBuf       [3]safe.Float32s
	fftBuf        [3]safe.Float32s
}

func (c *Conv) Init(input, output [3][]float32, size [3]int) {
	N := prod(size)
	for c := 0; c < 3; c++ {
		if len(output[c]) != N || len(input[c]) != N {
			panic(fmt.Errorf("xc.Conv.Init: inconsistent sizes"))
		}
	}
	c.size = size
	c.input = input
	c.output = output

	c.realBuf[0].Free()
	c.fftBuf[0].Free()
}

func prod(size [3]int) int {
	return size[0] * size[1] * size[2]
}
