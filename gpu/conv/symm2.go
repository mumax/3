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
	kern    [3][3][]float32     // Real-space kernel
	kernArr [3][3][][][]float32 // Real-space kernel
	fftKern [3][3][]float32     // FFT kernel on host
}

func (c *Symm2) init() {
	core.Debug("run")
	padded := c.kernSize

	// init device buffers
	c.deviceData3.init(c.size, c.kernSize)

	// init FFT plans
	c.stream = cu.StreamCreate()
	c.fwPlan = safe.FFT3DR2C(padded[0], padded[1], padded[2])
	c.fwPlan.SetStream(c.stream)
	c.bwPlan = safe.FFT3DC2R(padded[0], padded[1], padded[2])
	c.bwPlan.SetStream(c.stream)

	// init FFT kernel
	ffted := fftR2COutputSizeFloats(padded)
	realsize := ffted
	realsize[2] /= 2
	fwPlan := c.fwPlan
	output := safe.MakeComplex64s(fwPlan.OutputLen())
	defer output.Free()
	input := output.Float().Slice(0, fwPlan.InputLen())

	for i := 0; i < 3; i++ {
		for j := i; j < 3; j++ {
			input.CopyHtoD(c.kern[i][j])
			fwPlan.Exec(input, output)
			fwPlan.Stream().Synchronize() // !!
			c.fftKern[i][j] = make([]float32, prod(realsize))
			scaleRealParts(c.fftKern[i][j], output.Float(), 1/float32(fwPlan.InputLen()))
			// TODO: partially if low on mem.
			c.gpuFFTKern[i][j] = safe.MakeFloat32s(len(c.fftKern[i][j]))
			c.gpuFFTKern[i][j].CopyHtoD(c.fftKern[i][j])
		}
	}
}

func (c *Symm2) Run() {
	core.Debug("run")
	gpu.LockCudaThread()
	c.init()

	padded := c.kernSize
	offset := [3]int{0, 0, 0}
	for {

		// FW FFT
		for i := 0; i < 3; i++ {
			c.fftRBuf[i].MemsetAsync(0, c.stream) // copypad does NOT zero remainder.
			c.inlock[i].ReadNext(c.n)
			copyPad(c.fftRBuf[i], c.ioBuf[i], padded, c.size, offset, c.stream)
			c.inlock[i].ReadDone()
			c.fwPlan.Exec(c.fftRBuf[i], c.fftCBuf[i])
			c.stream.Synchronize()
		}

		// kern mul
		kernMulRSymm(c.fftCBuf,
			c.gpuFFTKern[0][0], c.gpuFFTKern[1][1], c.gpuFFTKern[2][2],
			c.gpuFFTKern[1][2], c.gpuFFTKern[0][2], c.gpuFFTKern[0][1],
			c.stream)
		c.stream.Synchronize()

		// BW FFT
		for i := 0; i < 3; i++ {
			c.bwPlan.Exec(c.fftCBuf[i], c.fftRBuf[i])
			c.outlock[i].WriteNext(c.n)
			copyPad(c.ioBuf[i], c.fftRBuf[i], c.size, padded, offset, c.stream)
			c.stream.Synchronize()
			c.outlock[i].WriteDone()
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
