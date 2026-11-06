package cuda

// Calculate stray field at a distance MFMLift above the sample.
// Based on conv_demag.go and conv_mfm.go

import (
	"github.com/mumax/3/data"
	"github.com/mumax/3/util"
)

// Stores the necessary state to perform FFT-accelerated convolution
type StrayFieldConvolution struct {
	inputSize        [3]int            // 3D size of the input/output data
	realKernSize     [3]int            // Size of kernel and logical FFT size
	fftKernLogicSize [3]int            // Logic size FFTed kernel, real parts only, we store less
	fftRBuf          [3]*data.Slice    // FFT input buf
	fftCBuf          [3]*data.Slice    // FFT output buf
	gpuFFTKern       [3][3]*data.Slice // FFT kernel on device
	fwPlan           fft3DR2CPlan      // Forward FFT (1 component)
	bwPlan           fft3DC2RPlan      // Backward FFT (1 component)
}

// Initializes a convolution to evaluate the demag field for the given mesh geometry.
func NewStray(inputSize, PBC [3]int, kernel [3][3]*data.Slice) *StrayFieldConvolution {
	c := new(StrayFieldConvolution)
	c.inputSize = inputSize
	c.realKernSize = kernel[X][X].Size()
	c.init(kernel)
	return c
}

// Calculate the stray field of m * vol * Msat, store result in outp.
//
//	m:    magnetization normalized to unit length
//	vol:  unitless mask used to scale m's length, may be nil
//	Msat: saturation magnetization in A/m
//	B:    resulting demag field, in Tesla
func (c *StrayFieldConvolution) Exec(B, m, vol *data.Slice, Msat MSlice) {
	util.Argument(B.Size() == c.inputSize && m.Size() == c.inputSize)
	for i := range 3 { // FW FFT comp i
		zero1_async(c.fftRBuf[i])
		copyPadMul(c.fftRBuf[i], m.Comp(i), vol, c.realKernSize, c.inputSize, Msat)
		c.fwPlan.ExecAsync(c.fftRBuf[i], c.fftCBuf[i])
	}

	// kern mul in Fourier space
	kernMulC3D_async(c.fftCBuf,
		c.gpuFFTKern[X][X], c.gpuFFTKern[Y][Y], c.gpuFFTKern[Z][Z],
		c.gpuFFTKern[Y][Z], c.gpuFFTKern[X][Z], c.gpuFFTKern[X][Y],
		c.fftKernLogicSize[X]/2, c.fftKernLogicSize[Y], c.fftKernLogicSize[Z]) // /2 since X-axis is used to store complex values

	for i := range 3 { // BW FFT comp i
		c.bwPlan.ExecAsync(c.fftCBuf[i], c.fftRBuf[i])
		copyUnPad(B.Comp(i), c.fftRBuf[i], c.inputSize, c.realKernSize)
	}
}

func (c *StrayFieldConvolution) init(realKern [3][3]*data.Slice) {
	// init device buffers
	c.fftKernLogicSize = fftR2COutputSizeFloats(c.realKernSize) // 2*X-dim to accommodate complex numbers in float slice
	c.fftCBuf[X] = NewSlice(1, c.fftKernLogicSize)
	c.fftCBuf[Y] = NewSlice(1, c.fftKernLogicSize)
	c.fftCBuf[Z] = NewSlice(1, c.fftKernLogicSize)

	c.fftRBuf[X] = NewSlice(1, c.realKernSize)
	c.fftRBuf[Y] = NewSlice(1, c.realKernSize)
	c.fftRBuf[Z] = NewSlice(1, c.realKernSize)

	// init FFT plans
	c.fwPlan = newFFT3DR2C(c.realKernSize[X], c.realKernSize[Y], c.realKernSize[Z])
	c.bwPlan = newFFT3DC2R(c.realKernSize[X], c.realKernSize[Y], c.realKernSize[Z])

	// init FFT kernel
	for i := range 3 {
		for j := range 3 {
			if realKern[i][j] == nil { // ignore 0's
				continue
			}
			zero1_async(c.fftRBuf[j])
			data.Copy(c.fftRBuf[j], realKern[i][j])
			c.fwPlan.ExecAsync(c.fftRBuf[j], c.fftCBuf[j])
			scale := 1 / float32(c.fwPlan.InputLen()) // 1/ instead of 2/ since no symmetries are assumed
			c.gpuFFTKern[i][j] = NewSlice(1, c.fftKernLogicSize)
			zero1_async(c.gpuFFTKern[i][j])
			Madd2(c.gpuFFTKern[i][j], c.gpuFFTKern[i][j], c.fftCBuf[j], 0, scale) // gpuFFTKern = scale*fftCBuf
		}
	}
}

func (c *StrayFieldConvolution) Free() {
	if c == nil {
		return
	}
	c.inputSize = [3]int{}
	c.realKernSize = [3]int{}
	for i := range 3 {
		c.fftCBuf[i].Free()
		c.fftRBuf[i].Free()
		c.fftCBuf[i] = nil
		c.fftRBuf[i] = nil

		for j := range 3 {
			c.gpuFFTKern[i][j].Free()
			c.gpuFFTKern[i][j] = nil
		}
		c.fwPlan.Free()
		c.bwPlan.Free()

		cudaCtx.SetCurrent()
	}
}
