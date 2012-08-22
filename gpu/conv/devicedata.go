package conv

import (
	"github.com/barnex/cuda4/safe"
	"nimble-cube/core"
	"nimble-cube/gpu"
)

// common 3-component device buffers
type deviceData3 struct {
	realBuf    [3]safe.Float32s    // gpu buffer for real-space, unpadded input/output data
	fftRBuf    [3]safe.Float32s    // Real ("input") buffers for FFT, shares underlying storage with fftCBuf
	fftCBuf    [3]safe.Complex64s  // Complex ("output") for FFT, shares underlying storage with fftRBuf
	gpuFFTKern [3][3]safe.Float32s // FFT kernel on device: TODO: xfer if needed
}

func(c*deviceData3)init(inputSize, kernelSize [3]int){
	for i := 0; i < 3; i++ {
		c.realBuf[i] = safe.MakeFloat32s(prod(c.size))
		c.fftCBuf[i] = safe.MakeComplex64s(c.fwPlan[i].OutputLen())
		c.fftRBuf[i] = c.fftCBuf[i].Float().Slice(0, c.fwPlan[i].InputLen())
	}
}
