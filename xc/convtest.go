package xc

import (
	"math"
	//"math/rand"
	"github.com/barnex/cuda4/safe"
	"nimble-cube/core"
)

// Run self-test and report estimated relative RMS error.
func (c *Conv) Test() {
	N := c.n
	var input [3][]float32

	for i := 0; i < 3; i++ {
		for j := range c.input[i] {
			c.input[i][j] = 1 //2*rand.Float32() - 1
			input[i] = make([]float32, N)
			copy(input[i], c.input[i])
		}
	}
	for i := 0; i < 3; i++ {
		core.Debug("input:", i, core.Format(safe.Reshape3DFloat32(input[i], c.size[0], c.size[1], c.size[2])))
	}

	//c.Push(N / 2)
	c.Push(N)
	//c.Pull(N / 2)
	c.Pull(N)
	//c.Push(N)
	//c.Pull(N)

	rms := 0.0
	for i := 0; i < 3; i++ {
		for j := range c.input[i] {
			rms += sqr(float64(input[i][j]) - float64(c.output[i][j])/float64(N))
		}
	}
	rms = math.Sqrt(rms / float64(3*N))

	for i := 0; i < 3; i++ {
		core.Debug("output:", i, core.Format(safe.Reshape3DFloat32(c.output[i], c.size[0], c.size[1], c.size[2])))
	}
	core.Log("RMS fft error:", rms)
}

func sqr(x float64) float64 {
	return x * x
}
