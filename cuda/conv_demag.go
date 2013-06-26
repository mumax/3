package cuda

import (
	"code.google.com/p/mx3/data"
	"code.google.com/p/mx3/mag"
	"github.com/barnex/cuda5/cu"
)

// Stores the necessary state to perform FFT-accelerated convolution
// with magnetostatic kernel (or other kernel of same symmetry).
type DemagConvolution struct {
	size        [3]int            // 3D size of the input/output data
	kernSize    [3]int            // Size of kernel and logical FFT size.
	fftKernSize [3]int            // Size of real, FFTed kernel
	n           int               // product of size
	fftRBuf     [3]*data.Slice    // FFT input buf for FFT, shares storage with fftCBuf.
	fftCBuf     [3]*data.Slice    // FFT output buf, shares storage with fftRBuf
	gpuFFTKern  [3][3]*data.Slice // FFT kernel on device
	fwPlan      fft3DR2CPlan      // Forward FFT (1 component)
	bwPlan      fft3DC2RPlan      // Backward FFT (1 component)
	kern        [3][3]*data.Slice // Real-space kernel (host)
	FFTMesh     data.Mesh         // mesh of FFT m
	stream      cu.Stream         // Stream for FFT plans
}

func (c *DemagConvolution) init() {
	{ // init FFT plans
		padded := c.kernSize
		c.stream = cu.StreamCreate()
		c.fwPlan = newFFT3DR2C(padded[0], padded[1], padded[2], c.stream)
		c.bwPlan = newFFT3DC2R(padded[0], padded[1], padded[2], c.stream)
	}

	{ // init device buffers
		// 2D re-uses fftBuf[1] as fftBuf[0], 3D needs all 3 fftBufs.
		for i := 1; i < 3; i++ {
			c.fftCBuf[i] = makeFloats(fftR2COutputSizeFloats(c.kernSize))
		}
		if c.is3D() {
			c.fftCBuf[0] = makeFloats(fftR2COutputSizeFloats(c.kernSize))
		} else {
			c.fftCBuf[0] = c.fftCBuf[1]
		}
		for i := 0; i < 3; i++ {
			c.fftRBuf[i] = c.fftCBuf[i].Slice(0, prod(c.kernSize))
		}
	}

	if c.is2D() {
		c.initFFTKern2D()
	} else {
		c.initFFTKern3D()
	}
}

func (c *DemagConvolution) initFFTKern3D() {
	padded := c.kernSize
	ffted := fftR2COutputSizeFloats(padded)
	realsize := ffted
	realsize[2] /= 2
	c.fftKernSize = realsize
	halfkern := realsize
	fwPlan := c.fwPlan
	output := c.fftCBuf[0]
	input := c.fftRBuf[0]

	// upper triangular part
	fftKern := data.NewSlice(1, data.NewMesh(halfkern[0], halfkern[1], halfkern[2], 1, 1, 1))
	for i := 0; i < 3; i++ {
		for j := i; j < 3; j++ {
			if c.kern[i][j] != nil { // ignore 0's
				data.Copy(input, c.kern[i][j])
				fwPlan.Exec(input, output)
				scaleRealParts(fftKern, output.Slice(0, prod(halfkern)*2), 1/float32(fwPlan.InputLen()))
				c.gpuFFTKern[i][j] = GPUCopy(fftKern)
			}
		}
	}
}

// Initialize GPU FFT kernel for 2D.
// Only the non-redundant parts are stored on the GPU.
func (c *DemagConvolution) initFFTKern2D() {
	padded := c.kernSize
	ffted := fftR2COutputSizeFloats(padded)
	realsize := ffted
	realsize[2] /= 2
	c.fftKernSize = realsize
	halfkern := realsize
	halfkern[1] = halfkern[1]/2 + 1
	fwPlan := c.fwPlan
	output := c.fftCBuf[0]
	input := c.fftRBuf[0]

	// upper triangular part
	fftKern := data.NewSlice(1, data.NewMesh(halfkern[0], halfkern[1], halfkern[2], 1, 1, 1))
	for i := 0; i < 3; i++ {
		for j := i; j < 3; j++ {
			if c.kern[i][j] != nil { // ignore 0's
				data.Copy(input, c.kern[i][j])
				fwPlan.Exec(input, output)
				scaleRealParts(fftKern, output.Slice(0, prod(halfkern)*2), 1/float32(fwPlan.InputLen()))
				c.gpuFFTKern[i][j] = GPUCopy(fftKern)
			}
		}
	}
}

// Calculate the demag field of m * vol * Bsat, store result in B.
// 	m:    magnetization normalized to unit length
// 	vol:  unitless mask used to scale m's length, may be nil
// 	Bsat: saturation magnetization in Tesla
// 	B:    resulting demag field, in Tesla
func (c *DemagConvolution) Exec(B, m *data.Slice, Bsat LUTPtr, regions *Bytes) {
	if c.is2D() {
		c.exec2D(B, m, Bsat, regions)
	} else {
		c.exec3D(B, m, Bsat, regions)
	}
}

// zero 1-component slice
func zero1(dst *data.Slice, str cu.Stream) {
	cu.MemsetD32Async(cu.DevicePtr(dst.DevPtr(0)), 0, int64(dst.Len()), str)
}

// forward FFT component i
func (c *DemagConvolution) fwFFT(i int, inp *data.Slice, Bsat LUTPtr, regions *Bytes) {
	zero1(c.fftRBuf[i], c.stream)
	in := inp.Comp(i)
	copyPadMul(c.fftRBuf[i], in, c.kernSize, c.size, Bsat, regions, c.stream)
	c.fwPlan.ExecAsync(c.fftRBuf[i], c.fftCBuf[i])
}

// backward FFT component i
func (c *DemagConvolution) bwFFT(i int, outp *data.Slice) {
	c.bwPlan.ExecAsync(c.fftCBuf[i], c.fftRBuf[i])
	out := outp.Comp(i)
	copyUnPad(out, c.fftRBuf[i], c.size, c.kernSize, c.stream)
}

// forward FFT of magnetization one component.
// returned slice is valid until next FFT or convolution
func (c *DemagConvolution) FFT(m *data.Slice, comp int, Bsat LUTPtr, regions *Bytes) *data.Slice {
	c.fwFFT(comp, m, Bsat, regions)
	return c.fftCBuf[comp]
}

func (c *DemagConvolution) exec3D(outp, inp *data.Slice, Bsat LUTPtr, regions *Bytes) {
	for i := 0; i < 3; i++ { // FW FFT
		c.fwFFT(i, inp, Bsat, regions)
	}

	// kern mul
	N0, N1, N2 := c.fftKernSize[0], c.fftKernSize[1], c.fftKernSize[2]
	kernMulRSymm3D(c.fftCBuf,
		c.gpuFFTKern[0][0], c.gpuFFTKern[1][1], c.gpuFFTKern[2][2],
		c.gpuFFTKern[1][2], c.gpuFFTKern[0][2], c.gpuFFTKern[0][1],
		N0, N1, N2, c.stream)

	for i := 0; i < 3; i++ { // BW FFT
		c.bwFFT(i, outp)
	}
	c.stream.Synchronize()
}

func (c *DemagConvolution) exec2D(outp, inp *data.Slice, Bsat LUTPtr, regions *Bytes) {
	// Convolution is separated into
	// a 1D convolution for x and a 2D convolution for yz.
	// So only 2 FFT buffers are needed at the same time.

	c.fwFFT(0, inp, Bsat, regions) // FFT x

	// kern mul X
	N1, N2 := c.fftKernSize[1], c.fftKernSize[2]
	kernMulRSymm2Dx(c.fftCBuf[0], c.gpuFFTKern[0][0], N1, N2, c.stream)

	c.bwFFT(0, outp) // bw FFT x

	for i := 1; i < 3; i++ { // FW FFT yz
		c.fwFFT(i, inp, Bsat, regions)
	}

	// kern mul yz
	kernMulRSymm2Dyz(c.fftCBuf[1], c.fftCBuf[2],
		c.gpuFFTKern[1][1], c.gpuFFTKern[2][2], c.gpuFFTKern[1][2],
		N1, N2, c.stream)

	for i := 1; i < 3; i++ { // BW FFT yz
		c.bwFFT(i, outp)
	}

	c.stream.Synchronize()
}

func (c *DemagConvolution) is2D() bool {
	return c.size[0] == 1
}

func (c *DemagConvolution) is3D() bool {
	return !c.is2D()
}

// Initializes a demag convolution for the given mesh geometry and magnetostatic kernel.
func newConvolution(mesh *data.Mesh, kernel [3][3]*data.Slice) *DemagConvolution {
	size := mesh.Size()
	c := new(DemagConvolution)
	c.size = size
	c.kern = kernel
	c.n = prod(size)
	c.kernSize = kernel[0][0].Mesh().Size()
	c.init()
	c.initMesh()
	testConvolution(c, mesh)
	c.freeKern()
	return c
}

// release the real-space kernel so the host memory can be reclaimed by GC.
func (c *DemagConvolution) freeKern() {
	for i := range c.kern {
		for j := range c.kern[i] {
			c.kern[i][j] = nil
		}
	}
}

func (c *DemagConvolution) initMesh() {
	n := fftR2COutputSizeFloats(c.kernSize)
	cell := c.kern[0][0].Mesh().CellSize()
	c.FFTMesh = *data.NewMesh(n[0], n[1], n[2], 1/cell[0], 1/cell[1], 1/cell[2])
	c.FFTMesh.Unit = "/m"
}

// Default accuracy setting for demag kernel.
const DEFAULT_KERNEL_ACC = 6

// Initializes a convolution to evaluate the demag field for the given mesh geometry.
func NewDemag(mesh *data.Mesh) *DemagConvolution {
	k := mag.BruteKernel(mesh, DEFAULT_KERNEL_ACC)
	return newConvolution(mesh, k)
}
