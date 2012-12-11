package gpu

import (
	"code.google.com/p/mx3/core"
)

// common data for all convolutions
type hostData struct {
	size          [3]int              // 3D size of the input/output data
	kernSize      [3]int              // Size of kernel and logical FFT size.
	n             int                 // product of size
	input, output [3][]float32        // input/output as contiguous lists, 3 component vectors
	inArr, outArr [3][][][]float32    // input/output as 3D array.
	kern          [3][3][]float32     // Real-space kernel
	kernArr       [3][3][][][]float32 // Real-space kernel
	fftKern       [3][3][]float32     // FFT kernel on host
}

// Input data.
func (c *hostData) Input() [3][][][]float32 {
	return c.inArr
}

// Output data.
func (c *hostData) Output() [3][][][]float32 {
	return c.outArr
}

// Convolution kernel.
func (c *hostData) Kernel() [3][3][][][]float32 {
	return c.kernArr
}

// initialize host arrays and check sizes.
func (c *hostData) init(size [3]int, kernel [3][3][][][]float32) {

	c.size = size
	c.n = core.Prod(c.size)
	c.inArr = core.MakeVectors(size)
	c.outArr = core.MakeVectors(size)
	c.input = core.Contiguous3(c.inArr)
	c.output = core.Contiguous3(c.outArr)
	for i := 0; i < 3; i++ {
		for j := 0; j < 3; j++ {
			if kernel[i][j] != nil {
				c.kern[i][j] = core.Contiguous(kernel[i][j])
			}
		}
	}
	c.kernSize = core.SizeOf(kernel[0][0])
	c.kernArr = kernel
}

// Page-lock host arrays if applicable.
// Should be run in CUDA locked thread.
func (c *hostData) initPageLock() {
	for i := 0; i < 3; i++ {
		MemHostRegister(c.input[i])
		MemHostRegister(c.output[i])
	}
}
