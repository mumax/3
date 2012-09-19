package conv

import (
	"github.com/barnex/cuda4/cu"
	"github.com/barnex/cuda4/safe"
	"nimble-cube/core"
	"nimble-cube/gpu"
)

type Symm2 struct {
	size     [3]int // 3D size of the input/output data
	kernSize [3]int // Size of kernel and logical FFT size.
	n        int    // product of size
	deviceData3
	inlock  [3]*core.RMutex
	fwPlan  safe.FFT3DR2CPlan
	bwPlan  safe.FFT3DC2RPlan
	outlock [3]*core.RWMutex
	stream  cu.Stream
}

func (c *Symm2) Run() {
	core.Debug("run")
	gpu.LockCudaThread()
	c.deviceData3.init(c.size, c.kernSize)
	// init fft

	for {

		for i := 0; i < 3; i++ {
			c.inlock[i].ReadNext(c.n)

			padded := c.kernSize
			offset := [3]int{0, 0, 0}
			c.fftRBuf[i].MemsetAsync(0, c.stream) // copypad does NOT zero remainder.
			copyPad(c.fftRBuf[i], c.ioBuf[i], padded, c.size, offset, c.stream)
			c.fwPlan.Exec(c.fftRBuf[i], c.fftCBuf[i])
			c.stream.Synchronize()

			c.inlock[i].ReadDone()
		}
	}
}

func NewSymm2(size [3]int, kernel [3][3][][][]float32) *Symm2 {
	c := new(Symm2)
	c.size = size
	c.n = prod(size)
	c.kernSize = core.SizeOf(kernel[0][0])

	return c
}
