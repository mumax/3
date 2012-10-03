package conv

import (
	"github.com/barnex/cuda4/cu"
	"github.com/barnex/cuda4/safe"
	"nimble-cube/core"
	"nimble-cube/gpu"
)

type Symm2 struct {
	size       [3]int              // 3D size of the input/output data
	kernSize   [3]int              // Size of kernel and logical FFT size.
	n          int                 // product of size
	input      [3]gpu.RChan        // TODO: fuse with input
	output     [3]gpu.Chan         // TODO: fuse with output
	fftRBuf    [3]safe.Float32s    // Real ("input") buffers for FFT, shares underlying storage with fftCBuf
	fftCBuf    [3]safe.Complex64s  // Complex ("output") for FFT, shares underlying storage with fftRBuf
	gpuFFTKern [3][3]safe.Float32s // FFT kernel on device: TODO: xfer if needed
	fwPlan     safe.FFT3DR2CPlan   // Forward FFT (1 component)
	bwPlan     safe.FFT3DC2RPlan   // Backward FFT (1 component)
	stream     cu.Stream           // 
	kern       [3][3][]float32     // Real-space kernel
	kernArr    [3][3][][][]float32 // Real-space kernel
	fftKern    [3][3][]float32     // FFT kernel on host
}

func (c *Symm2) init() {
	core.Log("initializing 2D symmetric convolution")
	padded := c.kernSize

	{ // init FFT plans
		c.stream = cu.StreamCreate()
		c.fwPlan = safe.FFT3DR2C(padded[0], padded[1], padded[2])
		c.fwPlan.SetStream(c.stream)
		c.bwPlan = safe.FFT3DC2R(padded[0], padded[1], padded[2])
		c.bwPlan.SetStream(c.stream)
	}

	{ // init FFT kernel
		ffted := fftR2COutputSizeFloats(padded)
		realsize := ffted
		realsize[2] /= 2
		fwPlan := c.fwPlan
		output := safe.MakeComplex64s(fwPlan.OutputLen())
		input := output.Float().Slice(0, fwPlan.InputLen())

		// upper triangular part
		for i := 0; i < 3; i++ {
			for j := i; j < 3; j++ {
				if c.kern[i][j] != nil { // ignore 0's
					input.CopyHtoD(c.kern[i][j])
					fwPlan.Exec(input, output)
					fwPlan.Stream().Synchronize() // !!
					c.fftKern[i][j] = make([]float32, prod(realsize))
					scaleRealParts(c.fftKern[i][j], output.Float(), 1/float32(fwPlan.InputLen()))
					c.gpuFFTKern[i][j] = safe.MakeFloat32s(len(c.fftKern[i][j]))
					c.gpuFFTKern[i][j].CopyHtoD(c.fftKern[i][j])
					core.Printf("% 6f", core.Reshape(c.fftKern[i][j], realsize))
				}
			}
		}
		output.Free()
	}

	{ // init device buffers
		for i := 0; i < 3; i++ {
			c.fftCBuf[i] = safe.MakeComplex64s(prod(fftR2COutputSizeFloats(c.kernSize)) / 2)
			c.fftRBuf[i] = c.fftCBuf[i].Float().Slice(0, prod(c.kernSize))
		}
	}
}

func (c *Symm2) Run() {
	core.Log("running symmetric 2D convolution")
	gpu.LockCudaThread()
	c.init()

	padded := c.kernSize
	offset := [3]int{0, 0, 0}
	for {

		// FW FFT
		for i := 0; i < 3; i++ {
			c.input[i].ReadNext(c.n)

			c.fftRBuf[i].MemsetAsync(0, c.stream) // copypad does NOT zero remainder.
			copyPad(c.fftRBuf[i], c.input[i].UnsafeData(), padded, c.size, offset, c.stream)
			c.fwPlan.Exec(c.fftRBuf[i], c.fftCBuf[i])
			c.stream.Synchronize()

			c.input[i].ReadDone()
		}

		// kern mul
		kernMulRSymm(c.fftCBuf,
			c.gpuFFTKern[0][0], c.gpuFFTKern[1][1], c.gpuFFTKern[2][2],
			c.gpuFFTKern[1][2], c.gpuFFTKern[0][2], c.gpuFFTKern[0][1],
			c.stream)
		c.stream.Synchronize()

		// BW FFT
		for i := 0; i < 3; i++ {
			c.output[i].WriteNext(c.n)

			c.bwPlan.Exec(c.fftCBuf[i], c.fftRBuf[i])
			copyPad(c.output[i].UnsafeData(), c.fftRBuf[i], c.size, padded, offset, c.stream)
			c.stream.Synchronize()

			c.output[i].WriteDone()
		}
	}
}

func NewSymm2(size [3]int, kernel [3][3][][][]float32, input [3]gpu.RChan, output [3]gpu.Chan) *Symm2 {
	core.Assert(size[0] == 1) // 2D not yet supported
	c := new(Symm2)
	c.size = size
	c.kernArr = kernel
	for i := 0; i < 3; i++ {
		for j := 0; j < 3; j++ {
			if kernel[i][j] != nil {
				c.kern[i][j] = core.Contiguous(kernel[i][j])
			}
		}
	}
	c.n = prod(size)
	c.kernSize = core.SizeOf(kernel[0][0])
	c.input = input
	c.output = output

	return c
	// TODO: self-test
}
