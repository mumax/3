package xc

import (
	"fmt"
)

type Conv struct {
	size          [3]int
	input, output [3][]float32
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
}

func prod(size [3]int) int {
	return size[0] * size[1] * size[2]
}
