package xc

import (
	"fmt"
	"github.com/barnex/cuda4/safe"
	"nimble-cube/core"
)

type Conv struct {
	size          [3]int
	input, output [3][]float32
	realBuf       [3]safe.Float32s
	fftBuf        [3]safe.Float32s

	fwPlan safe.FFT3DR2CPlan
	bwPlan safe.FFT3DC2RPlan
}

func NewConv(input, output [3][]float32, size [3]int) *Conv {
	c := new(Conv)
	N := prod(size)
	for c := 0; c < 3; c++ {
		if len(output[c]) != N || len(input[c]) != N {
			panic(fmt.Errorf("xc.Conv.Init: inconsistent sizes"))
		}
	}
	c.size = size
	c.input = input
	c.output = output
	go c.run()
	return c
}

func (c *Conv) run() {
	core.LockCudaThread()
	core.Debug("xc.Conv.run")

	// continue initialization here, inside locked CUDA thread
	c.init()

}

func (c *Conv) init() {
	core.Debug("xc.Conv.init")
	N := prod(c.size)

	// don't leak on 2nd init
	c.realBuf[0].Free()
	c.fftBuf[0].Free()

	r := safe.MakeFloat32s(3 * N)
	c.realBuf = [3]safe.Float32s{r.Slice(0*N, 1*N), r.Slice(1*N, 2*N), r.Slice(2*N, 3*N)}

	padded := PadSize(c.size)
	ffted := FFTR2COutputSizeFloats(padded)
	realsize := ffted
	realsize[2] /= 2
}

func prod(size [3]int) int {
	return size[0] * size[1] * size[2]
}

// Zero-padded size.
func PadSize(size [3]int) [3]int {
	padded := [3]int{
		size[0] * 2,
		size[1] * 2,
		size[2] * 2}
	if padded[0] == 2 {
		padded[0] = 1 // no need to pad 1 layer thickness
	}
	return padded
}

func FFTR2COutputSizeFloats(logicSize [3]int) [3]int {
	return [3]int{logicSize[0], logicSize[1], logicSize[2] + 2}
}
