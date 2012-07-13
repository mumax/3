package xc

import (
	"fmt"
	//"github.com/barnex/cuda4/safe"
	"math"
	"math/rand"
	"nimble-cube/core"
)

// Run self-test and report estimated relative RMS error
// on forward+backward transform on data of magnitude order 1.
func (c *Conv1) Test() {
	N := c.n

	// make random input data between -1 and 1
	for i := 0; i < 3; i++ {
		for j := range c.input[i] {
			c.input[i][j] = 2*rand.Float32() - 1
		}
	}

	{
		c.noKernMul = true // don't do kernel multiplication, only fw+bw transform

		c.Push(N)
		c.Pull(N - 1)
		c.Pull(N)
		c.checkError()

		c.Push(N / 2)
		c.Push(N)
		c.Pull(N / 4)
		c.Pull(N - 1)
		c.Pull(N)
		c.checkError()

		c.noKernMul = false
	}

	// Set i/o arrays back to 0
	for i := 0; i < 3; i++ {
		for j := range c.input[i] {
			c.input[i][j] = 0
			c.output[i][j] = 0
		}
	}
}

const FFT_TOLERANCE = 1e-5 // panic if RMS error of fw+bw transform on random data is larger than this.

// Check if the rms error introduced by fw+bw transform is < FFT_TOLERANCE
func (c *Conv1) checkError() {
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
		// Something is wrong with the test
		panic(fmt.Errorf("FFT RMS error: %v: too good to be true", rms))
	}
}

func sqr(x float64) float64 {
	return x * x
}
