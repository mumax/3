package conv

import (
	"github.com/barnex/cuda4/safe"
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
