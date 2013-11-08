package cuda

import (
	"github.com/mumax/3/data"
	"github.com/mumax/3/mag"
	"github.com/mumax/3/util"
	"log"
)

// Stores the necessary state to perform FFT-accelerated convolution
type MFMConvolution struct {
	size        [3]int         // 3D size of the input/output data
	kernSize    [3]int         // Size of kernel and logical FFT size.
	fftKernSize [3]int         // Size of real, FFTed kernel
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

	c.initFFTKern3D()
}

func (c *MFMConvolution) initFFTKern3D() {

	// size of FFT(kernel): store real parts only
	c.fftKernSize = fftR2COutputSizeFloats(c.kernSize)
	util.Assert(c.fftKernSize[X]%2 == 0)
	c.fftKernSize[X] /= 2

	// will store only 1/4 (symmetry), but not yet
	halfkern := c.fftKernSize

	output := c.fftCBuf
	input := c.fftRBuf

	fftKern := data.NewSlice(1, data.NewMesh(halfkern[X], halfkern[Y], halfkern[Z], 1, 1, 1)) // host
	for i := 0; i < 3; i++ {
		data.Copy(input, c.kern[i])
		c.fwPlan.Exec(input, output)
		scaleRealParts(fftKern, output.Slice(0, prod(halfkern)*2), 1/float32(c.fwPlan.InputLen()))
		log.Println(fftKern)
		c.gpuFFTKern[i] = GPUCopy(fftKern)
	}
}

func (c *MFMConvolution) Exec(B, m, vol *data.Slice, Bsat LUTPtr, regions *Bytes) {
	c.exec3D(B, m, vol, Bsat, regions)
}

func (c *MFMConvolution) exec3D(outp, inp, vol *data.Slice, Bsat LUTPtr, regions *Bytes) {

	for i := 0; i < 3; i++ {
		// fw fft
		zero1_async(c.fftRBuf)
		copyPadMul(c.fftRBuf, inp.Comp(i), vol, c.kernSize, c.size, Bsat, regions)
		c.fwPlan.ExecAsync(c.fftRBuf, c.fftCBuf)

		// kern mul Z
		Nx, Ny := c.fftKernSize[X], c.fftKernSize[Y]
		kernMulRSymm2Dz_async(c.fftCBuf, c.gpuFFTKern[i], Nx, Ny)

		c.bwPlan.ExecAsync(c.fftCBuf, c.fftRBuf)
		copyUnPad(outp.Comp(i), c.fftRBuf, c.size, c.kernSize)
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
