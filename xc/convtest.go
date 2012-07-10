package xc

import (
	"fmt"
	"github.com/barnex/cuda4/safe"
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
	for i := 0; i < 3; i++ {
		core.Debug("input:", i, core.Format(safe.Reshape3DFloat32(input[i], c.size[0], c.size[1], c.size[2])))
	}

	c.Push(N)
	c.Pull(N)
	c.checkError()

	c.Push(N / 2)
	c.Push(N)
	c.Pull(N / 4)
	c.Pull(N - 1)
	c.Pull(N)
	c.checkError()

}

const FFT_TOLERANCE = 1e-6

func (c *Conv) checkError() {
	NFFT := prod(PadSize(c.size))
	rms := 0.0
	for i := 0; i < 3; i++ {
		for j := range c.input[i] {
			rms += sqr(float64(c.input[i][j]) - float64(c.output[i][j])/float64(NFFT))
		}
	}
	rms = math.Sqrt(rms / float64(3*NFFT))

	core.Debug("RMS fft error:", rms)
	if rms > FFT_TOLERANCE {
		panic(fmt.Errorf("FFT RMS error: %v > tolerance (%v)", rms, FFT_TOLERANCE))
	}
	if rms == 0 {
		panic(fmt.Errorf("FFT RMS error: %v: too good to be true", rms))
	}
}

func sqr(x float64) float64 {
	return x * x
}
