package conv

import (
	"github.com/barnex/cuda4/safe"
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

// common 3-component device buffers
type deviceData3 struct {
	realBuf [3]safe.Float32s    // gpu buffer for real-space, unpadded input/output data
	fftRBuf [3]safe.Float32s    // Real ("input") buffers for FFT, shares underlying storage with fftCBuf
	fftCBuf [3]safe.Complex64s  // Complex ("output") for FFT, shares underlying storage with fftRBuf
	gpuKern [3][3]safe.Float32s // FFT kernel on device: TODO: xfer if needed
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

func fftR2COutputSizeFloats(logicSize [3]int) [3]int {
	return [3]int{logicSize[0], logicSize[1], 2 * (logicSize[2]/2 + 1)}
}
