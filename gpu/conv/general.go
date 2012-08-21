package conv

import (
	"github.com/barnex/cuda4/safe"
	"nimble-cube/core"
)

// General convolution, not optimized for specific cases.
type General struct {
	hostData    // sizes, host input/output/kernel arrays
	deviceData3 // device buffers
	fwPlan      safe.FFT3DR2CPlan
	bwPlan      safe.FFT3DC2RPlan
}

func NewGeneral(input_, output_ [3][][][]float32, kernel [3][3][][][]float32) *General {
	c := new(General)
	c.hostData.init(input_, output_, kernel)

	// need cuda thread lock from here on:
	c.hostData.initPageLock()
	c.initFFT()
	c.initFFTKern()

	return c
}

func (c *General) initFFT() {
	padded := c.kernSize
	//realsize := fftR2COutputSizeFloats(padded)
	c.fwPlan = safe.FFT3DR2C(padded[0], padded[1], padded[2])
	c.bwPlan = safe.FFT3DC2R(padded[0], padded[1], padded[2])
	// no streams set yet
}

func (c *General) initFFTKern() {

	realsize := c.kernSize
	reallen := prod(realsize)
	fftedsize := fftR2COutputSizeFloats(realsize)
	fftedlen := prod(fftedsize)

	fwPlan := c.fwPlan // could use any

	for i := 0; i < 3; i++ {
		for j := 0; j < 3; j++ {
			c.fftKern[i][j] = make([]float32, fftedlen)
			c.gpuKern[i][j] = safe.MakeFloat32s(fftedlen)
			c.gpuKern[i][j].Slice(0, reallen).CopyHtoD(c.kern[i][j])
			fwPlan.Exec(c.gpuKern[i][j].Slice(0, reallen), c.gpuKern[i][j].Complex())
			c.gpuKern[i][j].CopyHtoD(c.fftKern[i][j])
		}
	}
}
