package xc

import (
	"math"
	"math/rand"
	"nimble-cube/core"
)

// Run self-test and report estimated relative RMS error.
func (c *Conv) Test() {
	N := c.n
	var input [3][]float32

	for i := 0; i < 3; i++ {
		for j := range c.input[i] {
			c.input[i][j] = 2*rand.Float32() - 1
			input[i] = make([]float32, N)
			copy(input[i], c.input[i])
		}
	}

	c.Push(N)
	c.Pull()

	rms := 0.0
	for i := 0; i < 3; i++ {
		for j := range c.input[i] {
			rms += sqr(input[i][j] - c.output[i][j])
		}
	}
	rms = math.Sqrt(rms)

	core.Log("RMS fft error:", rms)
}

func sqr(x float32) float64 {
	return float64(x) * float64(x)
}
