package conv

import (
	"nimble-cube/core"
	"nimble-cube/gpu"
)

// common data for all convolutions
type hostData struct {
	size          [3]int          // 3D size of the input/output data
	kernSize      [3]int          // Size of kernel and logical FFT size.
	n             int             // product of size
	input, output [3][]float32    // input/output arrays, 3 component vectors
	kern          [3][3][]float32 // Real-space kernel
	fftKern       [3][3][]float32 // FFT kernel on host
}

// Size of the input and output arrays.
func (c *hostData) IOSize() [3]int {
	return c.size
}

// Size of the convolution kernel, in real space.
// This is also the logic size of the FFTs.
func (c *hostData) KernelSize() [3]int {
	return c.kernSize
}

// initialize host arrays and check sizes.
func (c *hostData) init(input_, output_ [3][][][]float32, kernel [3][3][][][]float32) {
	c.size = core.SizeOf(input_[0])
	c.n = core.Prod(c.size)
	for i := 0; i < 3; i++ {
		core.CheckEqualSize(core.SizeOf(input_[i]), c.size)
		core.CheckEqualSize(core.SizeOf(output_[i]), c.size)
		c.input[i] = core.Contiguous(input_[i])
		c.output[i] = core.Contiguous(output_[i])
		for j := 0; j < 3; j++ {
			if kernel[i][j] != nil {
				c.kern[i][j] = core.Contiguous(kernel[i][j])
			}
		}
	}
	c.kernSize = core.SizeOf(kernel[0][0])
}

// Page-lock host arrays if applicable.
// Should be run in CUDA locked thread.
func (c *hostData) initPageLock() {
	for i := 0; i < 3; i++ {
		gpu.MemHostRegister(c.input[i])
		gpu.MemHostRegister(c.output[i])
	}
}

// Output size of R2C FFT with given logic size, expressed in floats.
func fftR2COutputSizeFloats(logicSize [3]int) [3]int {
	return [3]int{logicSize[0], logicSize[1], 2 * (logicSize[2]/2 + 1)}
}
