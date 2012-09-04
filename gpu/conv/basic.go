package conv

import (
	"github.com/barnex/cuda4/safe"
	//"nimble-cube/core"
)

// Basic convolution, not optimized for specific cases.
// Also not concurrent.
// Kernel does not need to have any symmetry.
type Basic struct {
	hostData    // sizes, host input/output/kernel arrays
	deviceData3 // device buffers // could use just one ioBuf
	fwPlan      safe.FFT3DR2CPlan
	bwPlan      safe.FFT3DC2RPlan
}

// Execute the convolution.
func (c *Basic) Exec() {
	// Zero padding and forward FFTs.
	for i := 0; i < 3; i++ {
		//core.Println("input", i, "\n", core.Reshape(c.input[i], c.size))
		c.ioBuf[i].CopyHtoD(c.input[i])
		//core.Println("ioBuf", i, "\n", core.Reshape(c.ioBuf[i].Host(), c.size))
		stream0.Synchronize()
		c.copyPadIOBuf(i)
		//core.Println("fftRBuf", i, "\n", core.Reshape(c.fftRBuf[i].Host(), c.kernSize))
		stream0.Synchronize()
		c.fwPlan.Exec(c.fftRBuf[i], c.fftCBuf[i])
		//core.Println("fftCBuf", i, "\n", core.Reshape(c.fftCBuf[i].Float().Host(), c.fftKernelSizeFloats()))
		stream0.Synchronize()
	}

	// Kernel multiplication
	kernMulC(c.fftCBuf, c.gpuFFTKern, stream0)
	stream0.Synchronize()

	// Backward FFT and unpadding
	for i := 0; i < 3; i++ {
		//core.Println("fftCBuf", i, "\n", core.Reshape(c.fftCBuf[i].Float().Host(), c.fftKernelSizeFloats()))
		c.bwPlan.Exec(c.fftCBuf[i], c.fftRBuf[i])
		stream0.Synchronize()
		//core.Println("fftRBuf", i, "\n", core.Reshape(c.fftRBuf[i].Host(), c.kernSize))
		c.copyUnpadIOBuf(i)
		stream0.Synchronize()
		//core.Println("ioBuf", i, "\n", core.Reshape(c.ioBuf[i].Host(), c.size))
		c.ioBuf[i].CopyDtoH(c.output[i])
		stream0.Synchronize()
		//core.Println("output", i, "\n", core.Reshape(c.output[i], c.size))
	}
}

// Copy ioBuf[i] to fftRBuf[i], adding padding :-).
func (c *Basic) copyPadIOBuf(i int) {
	offset := [3]int{0, 0, 0}
	c.fftRBuf[i].Memset(0) // copypad does NOT zero remainder.
	stream0.Synchronize()
	copyPad(c.fftRBuf[i], c.ioBuf[i], c.kernSize, c.size, offset, stream0)
	stream0.Synchronize()
}

// Copy ioBuf[i] to fftRBuf[i], adding padding :-).
func (c *Basic) copyUnpadIOBuf(i int) {
	offset := [3]int{0, 0, 0}
	copyPad(c.ioBuf[i], c.fftRBuf[i], c.size, c.kernSize, offset, stream0)
	stream0.Synchronize()
}

// Size of the FFT'ed kernel expressed in number of floats.
// Real and Complex parts are stored.
func (c *Basic) fftKernelSizeFloats() [3]int {
	return fftR2COutputSizeFloats(c.kernSize)
	// kernel size is FFT logic size
}

// Initializes c.gpuFFTKern and c.fftKern
func (c *Basic) initFFTKern() {
	realsize := c.kernSize
	reallen := prod(realsize)
	fftedsize := fftR2COutputSizeFloats(realsize)
	fftedlen := prod(fftedsize)

	fwPlan := c.fwPlan // could use any

	norm := float32(1 / float64(prod(c.kernSize)))
	for i := 0; i < 3; i++ {
		for j := 0; j < 3; j++ {
			c.fftKern[i][j] = make([]float32, fftedlen)
			c.gpuFFTKern[i][j] = safe.MakeFloat32s(fftedlen)
			c.gpuFFTKern[i][j].Slice(0, reallen).CopyHtoD(scaled(c.kern[i][j], norm)) // scale could be on gpu...
			fwPlan.Exec(c.gpuFFTKern[i][j].Slice(0, reallen), c.gpuFFTKern[i][j].Complex())
			c.gpuFFTKern[i][j].CopyDtoH(c.fftKern[i][j])
		}
	}
}

func scaled(x []float32, s float32) []float32 {
	out := make([]float32, len(x))
	for i := range x {
		out[i] = x[i] * s
	}
	return out
}

// Initializes the FFT plans.
func (c *Basic) initFFT() {
	padded := c.kernSize
	//realsize := fftR2COutputSizeFloats(padded)
	c.fwPlan = safe.FFT3DR2C(padded[0], padded[1], padded[2])
	c.bwPlan = safe.FFT3DC2R(padded[0], padded[1], padded[2])
	// no streams set yet
}

func NewBasic(size [3]int, kernel [3][3][][][]float32) Conv {
	c := new(Basic)
	c.hostData.init(size, kernel)

	// need cuda thread lock from here on:
	c.hostData.initPageLock()
	c.initFFT()
	c.initFFTKern()
	c.deviceData3.init(c.size, c.kernSize)

	Test(c)

	return c
}
