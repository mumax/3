package cuda

import (
	"github.com/mumax/3/data"
	"github.com/mumax/3/mag"
)

// Stores the necessary state to perform FFT-accelerated convolution
type MFMConvolution struct {
	size        [3]int         // 3D size of the input/output data
	kernSize    [3]int         // Size of kernel and logical FFT size.
	fftKernSize [3]int         //
	fftRBuf     *data.Slice    // FFT input buf for FFT, shares storage with fftCBuf.
	fftCBuf     *data.Slice    // FFT output buf, shares storage with fftRBuf
	gpuFFTKern  [3]*data.Slice // FFT kernel on device
	fwPlan      fft3DR2CPlan   // Forward FFT (1 component)
	bwPlan      fft3DC2RPlan   // Backward FFT (1 component)
	kern        [3]*data.Slice // Real-space kernel (host)
}

func (c *MFMConvolution) init() {
	// init FFT plans
	padded := c.kernSize
	c.fwPlan = newFFT3DR2C(padded[X], padded[Y], padded[Z], stream0)
	c.bwPlan = newFFT3DC2R(padded[X], padded[Y], padded[Z], stream0)

	// init device buffers
	nc := fftR2COutputSizeFloats(c.kernSize)
	c.fftCBuf = makeFloats(nc)
	c.fftRBuf = c.fftCBuf.Slice(0, prod(c.kernSize))

	c.gpuFFTKern[X] = makeFloats(nc)
	c.gpuFFTKern[Y] = makeFloats(nc)
	c.gpuFFTKern[Z] = makeFloats(nc)

	c.initFFTKern3D()
}

func (c *MFMConvolution) initFFTKern3D() {

	c.fftKernSize = fftR2COutputSizeFloats(c.kernSize)

	for i := 0; i < 3; i++ {
		zero1_async(c.fftRBuf)
		data.Copy(c.fftRBuf, c.kern[i])
		c.fwPlan.Exec(c.fftRBuf, c.fftCBuf)
		scale := 2 / float32(c.fwPlan.InputLen()) // ??
		zero1_async(c.gpuFFTKern[i])
		Madd2(c.gpuFFTKern[i], c.gpuFFTKern[i], c.fftCBuf, 0, scale)
		//dbg("kern", i, c.gpuFFTKern[i].HostCopy())
	}
}

func (c *MFMConvolution) Exec(B, m, vol *data.Slice, Bsat LUTPtr, regions *Bytes) {
	c.exec3D(B, m, vol, Bsat, regions)
}

func (c *MFMConvolution) exec3D(outp, inp, vol *data.Slice, Bsat LUTPtr, regions *Bytes) {

	for i := 0; i < 3; i++ {
		//dbg("in", inp.Comp(i).HostCopy())
		zero1_async(c.fftRBuf)
		copyPadMul(c.fftRBuf, inp.Comp(i), vol, c.kernSize, c.size, Bsat, regions)
		c.fwPlan.ExecAsync(c.fftRBuf, c.fftCBuf)
		//dbg("fw FFT", c.fftCBuf.HostCopy())

		Nx, Ny := c.fftKernSize[X]/2, c.fftKernSize[Y] //   ??
		kernMulC_async(c.fftCBuf, c.gpuFFTKern[i], Nx, Ny)
		//dbg("mul", c.fftCBuf.HostCopy())

		c.bwPlan.ExecAsync(c.fftCBuf, c.fftRBuf)
		//dbg("bw FFT", c.fftCBuf.HostCopy())
		copyUnPad(outp.Comp(i), c.fftRBuf, c.size, c.kernSize)
		//dbg("out ", outp.Comp(i).HostCopy())
	}
}

// Initializes a convolution to evaluate the demag field for the given mesh geometry.
func NewMFM(mesh *data.Mesh, lift, tipsize float64) *MFMConvolution {
	k := mag.MFMKernel(mesh, lift, tipsize)
	size := mesh.Size()
	c := new(MFMConvolution)
	c.size = size
	c.kern = k
	c.kernSize = k[X].Mesh().Size()
	c.init()
	return c
}
