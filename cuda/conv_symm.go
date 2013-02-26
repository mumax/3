package cuda

import (
	"code.google.com/p/mx3/data"
	"code.google.com/p/mx3/util"
	"github.com/barnex/cuda5/cu"
)

type DemagConvolution struct {
	size        [3]int            // 3D size of the input/output data
	kernSize    [3]int            // Size of kernel and logical FFT size.
	fftKernSize [3]int            // Size of real, FFTed kernel
	n           int               // product of size
	fftRBuf     [3]*data.Slice    // FFT input buf for FFT, shares storage with fftCBuf.
	fftCBuf     [3]*data.Slice    // FFT output buf, shares storage with fftRBuf
	gpuFFTKern  [3][3]*data.Slice // FFT kernel on device: TODO: xfer if needed
	fwPlan      FFT3DR2CPlan      // Forward FFT (1 component)
	bwPlan      FFT3DC2RPlan      // Backward FFT (1 component)
	kern        [3][3]*data.Slice // Real-space kernel (host)
	stream      cu.Stream         // Stream for FFT plans
}

func (c *DemagConvolution) init() {
	{ // init FFT plans
		padded := c.kernSize
		c.stream = cu.StreamCreate()
		c.fwPlan = NewFFT3DR2C(padded[0], padded[1], padded[2], c.stream)
		c.bwPlan = NewFFT3DC2R(padded[0], padded[1], padded[2], c.stream)
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

func (c *DemagConvolution) Exec(outp *data.Slice, inp ...*data.Slice) {
	util.Argument(len(inp) == 1)
	if c.is2D() {
		c.exec2D(outp, inp[0])
	} else {
		c.exec3D(outp, inp[0])
	}
}

// zero 1-component slice
func zero1(dst *data.Slice, str cu.Stream) {
	cu.MemsetD32Async(cu.DevicePtr(dst.DevPtr(0)), 0, int64(dst.Len()), str)
}

func (c *DemagConvolution) exec3D(outp, inp *data.Slice) {
	padded := c.kernSize

	// FW FFT
	for i := 0; i < 3; i++ {
		zero1(c.fftRBuf[i], c.stream)
		in := inp.Comp(i)
		copyPad(c.fftRBuf[i], in, padded, c.size, c.stream)
		c.fwPlan.ExecAsync(c.fftRBuf[i], c.fftCBuf[i])
	}

	// kern mul
	N0, N1, N2 := c.fftKernSize[0], c.fftKernSize[1], c.fftKernSize[2] // TODO: rm these
	kernMulRSymm3D(c.fftCBuf,
		c.gpuFFTKern[0][0], c.gpuFFTKern[1][1], c.gpuFFTKern[2][2],
		c.gpuFFTKern[1][2], c.gpuFFTKern[0][2], c.gpuFFTKern[0][1],
		N0, N1, N2, c.stream)

	// BW FFT
	for i := 0; i < 3; i++ {
		c.bwPlan.ExecAsync(c.fftCBuf[i], c.fftRBuf[i])
		out := outp.Comp(i)
		copyPad(out, c.fftRBuf[i], c.size, padded, c.stream)
	}
	c.stream.Synchronize()
}

func (c *DemagConvolution) exec2D(outp, inp *data.Slice) {
	// Convolution is separated into
	// a 1D convolution for x and a 2D convolution for yz.
	// So only 2 FFT buffers are needed at the same time.

	// FFT x
	Memset(c.fftRBuf[0], 0)
	in := inp.Comp(0)
	padded := c.kernSize
	copyPad(c.fftRBuf[0], in, padded, c.size, c.stream)
	c.fwPlan.Exec(c.fftRBuf[0], c.fftCBuf[0])
	c.stream.Synchronize()

	// kern mul X
	N1, N2 := c.fftKernSize[1], c.fftKernSize[2] // TODO: rm these
	kernMulRSymm2Dx(c.fftCBuf[0], c.gpuFFTKern[0][0], N1, N2)

	// bw FFT x
	c.bwPlan.Exec(c.fftCBuf[0], c.fftRBuf[0])
	c.stream.Synchronize()
	out := outp.Comp(0)
	copyPad(out, c.fftRBuf[0], c.size, padded, c.stream)
	c.stream.Synchronize()

	// FW FFT yz
	for i := 1; i < 3; i++ {
		Memset(c.fftRBuf[i], 0)
		in := inp.Comp(i)
		copyPad(c.fftRBuf[i], in, padded, c.size, c.stream)
		c.fwPlan.Exec(c.fftRBuf[i], c.fftCBuf[i])
		c.stream.Synchronize()
	}

	// kern mul yz
	kernMulRSymm2Dyz(c.fftCBuf[1], c.fftCBuf[2],
		c.gpuFFTKern[1][1], c.gpuFFTKern[2][2], c.gpuFFTKern[1][2],
		N1, N2)

	// BW FFT yz
	for i := 1; i < 3; i++ {
		c.bwPlan.Exec(c.fftCBuf[i], c.fftRBuf[i])
		out := outp.Comp(i)
		c.stream.Synchronize()
		copyPad(out, c.fftRBuf[i], c.size, padded, c.stream)
		c.stream.Synchronize()
	}
}

func (c *DemagConvolution) is2D() bool {
	return c.size[0] == 1
}

func (c *DemagConvolution) is3D() bool {
	return !c.is2D()
}

func NewConvolution(mesh *data.Mesh, kernel [3][3]*data.Slice) *DemagConvolution {
	size := mesh.Size()
	c := new(DemagConvolution)
	c.size = size
	c.kern = kernel
	c.n = prod(size)
	c.kernSize = kernel[0][0].Mesh().Size()
	c.init()
	testConvolution(c, mesh)
	return c
}
