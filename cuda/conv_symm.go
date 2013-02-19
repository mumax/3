package cuda

import (
	"code.google.com/p/mx3/data"
	"github.com/barnex/cuda5/cu"
	"log"
)

type Symm2D struct {
	size        [3]int            // 3D size of the input/output data
	kernSize    [3]int            // Size of kernel and logical FFT size.
	fftKernSize [3]int            // Size of real, FFTed kernel
	n           int               // product of size
	input       *data.Reader      //
	output      *data.Quant       //
	fftRBuf     [3]*data.Slice    // FFT input buf for FFT, shares storage with fftCBuf.
	fftCBuf     [3]*data.Slice    // FFT output buf, shares storage with fftRBuf
	gpuFFTKern  [3][3]*data.Slice // FFT kernel on device: TODO: xfer if needed
	fwPlan      FFT3DR2CPlan      // Forward FFT (1 component)
	bwPlan      FFT3DC2RPlan      // Backward FFT (1 component)
	kern        [3][3]*data.Slice // Real-space kernel (host)
	inited      bool
}

func (c *Symm2D) init() {
	log.Println("initializing convolution")
	if c.inited {
		log.Panic("conv: already initialized")
	}
	c.inited = true

	{ // init FFT plans
		padded := c.kernSize
		stream := cu.StreamCreate()
		c.fwPlan = NewFFT3DR2C(padded[0], padded[1], padded[2], stream)
		c.bwPlan = NewFFT3DC2R(padded[0], padded[1], padded[2], stream)
	}

	{ // init device buffers
		// 2D re-uses fftBuf[1] as fftBuf[0], 3D needs all 3 fftBufs.
		for i := 1; i < 3; i++ {
			c.fftCBuf[i] = TryMakeFloats(fftR2COutputSizeFloats(c.kernSize))
		}
		if c.is3D() {
			c.fftCBuf[0] = TryMakeFloats(fftR2COutputSizeFloats(c.kernSize))
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

func (c *Symm2D) initFFTKern3D() {
	padded := c.kernSize
	ffted := fftR2COutputSizeFloats(padded)
	realsize := ffted
	realsize[2] /= 2
	c.fftKernSize = realsize
	halfkern := realsize
	fwPlan := c.fwPlan
	output := c.fftCBuf[0] //TryMakeFloats(fftR2COutputSizeFloats(c.kernSize))
	input := c.fftRBuf[0]  //output.Slice(0, fwPlan.InputLen())

	// upper triangular part
	fftKern := data.NewSlice(1, data.NewMesh(halfkern[0], halfkern[1], halfkern[2], 1, 1, 1))
	for i := 0; i < 3; i++ {
		for j := i; j < 3; j++ {
			if c.kern[i][j] != nil { // ignore 0's
				data.Copy(input, c.kern[i][j]) //input.CopyHtoD(c.kern[i][j])
				fwPlan.Exec(input, output)
				scaleRealParts(fftKern, output.Slice(0, prod(halfkern)*2), 1/float32(fwPlan.InputLen()))
				c.gpuFFTKern[i][j] = GPUCopy(fftKern)
			}
		}
	}
}

// Initialize GPU FFT kernel for 2D.
// Only the non-redundant parts are stored on the GPU.
func (c *Symm2D) initFFTKern2D() {
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

func (c *Symm2D) Run() {
	log.Println("running convolution")
	LockThread()

	for {
		c.Exec()
	}
}

func (c *Symm2D) Exec() {
	if c.is2D() {
		c.exec2D()
	} else {
		c.exec3D()
	}
}

func (c *Symm2D) exec3D() {
	// FW FFT
	for i := 0; i < 3; i++ {
		inc := c.input.Comp(i)
		in := inc.ReadNext(c.n)
		Memset(c.fftRBuf[i], 0)
		copyPad(c.fftRBuf[i], in)
		inc.ReadDone()
		c.fwPlan.Exec(c.fftRBuf[i], c.fftCBuf[i])
	}

	// kern mul
	N0, N1, N2 := c.fftKernSize[0], c.fftKernSize[1], c.fftKernSize[2] // TODO: rm these
	kernMulRSymm3D(c.fftCBuf,
		c.gpuFFTKern[0][0], c.gpuFFTKern[1][1], c.gpuFFTKern[2][2],
		c.gpuFFTKern[1][2], c.gpuFFTKern[0][2], c.gpuFFTKern[0][1],
		N0, N1, N2)

	// BW FFT
	for i := 0; i < 3; i++ {
		outc := c.output.Comp(i)
		c.bwPlan.Exec(c.fftCBuf[i], c.fftRBuf[i])
		out := outc.WriteNext(c.n)
		copyPad(out, c.fftRBuf[i]) //, c.size, padded)
		outc.WriteDone()
	}
}

func (c *Symm2D) exec2D() {
	// Convolution is separated into
	// a 1D convolution for x and a 2D convolution for yz.
	// So only 2 FFT buffers are needed at the same time.

	// FFT x
	Memset(c.fftRBuf[0], 0)
	inc := c.input.Comp(0)
	in := inc.ReadNext(c.n)
	copyPad(c.fftRBuf[0], in) //, padded, c.size)
	inc.ReadDone()
	c.fwPlan.Exec(c.fftRBuf[0], c.fftCBuf[0])

	// kern mul X
	N1, N2 := c.fftKernSize[1], c.fftKernSize[2] // TODO: rm these
	kernMulRSymm2Dx(c.fftCBuf[0], c.gpuFFTKern[0][0], N1, N2)
	//c.stream.Synchronize()

	// bw FFT x
	c.bwPlan.Exec(c.fftCBuf[0], c.fftRBuf[0])
	outc := c.output.Comp(0)
	out := outc.WriteNext(c.n)
	copyPad(out, c.fftRBuf[0]) //, c.size, padded)
	outc.WriteDone()

	// FW FFT yz
	for i := 1; i < 3; i++ {
		Memset(c.fftRBuf[i], 0)
		inc := c.input.Comp(i)
		in := inc.ReadNext(c.n)
		copyPad(c.fftRBuf[i], in) //, padded, c.size)
		inc.ReadDone()
		c.fwPlan.Exec(c.fftRBuf[i], c.fftCBuf[i])
	}

	// kern mul yz
	kernMulRSymm2Dyz(c.fftCBuf[1], c.fftCBuf[2],
		c.gpuFFTKern[1][1], c.gpuFFTKern[2][2], c.gpuFFTKern[1][2],
		N1, N2)

	// BW FFT yz
	for i := 1; i < 3; i++ {
		c.bwPlan.Exec(c.fftCBuf[i], c.fftRBuf[i])
		outc := c.output.Comp(i)
		out := outc.WriteNext(c.n)
		copyPad(out, c.fftRBuf[i]) //, c.size, padded)
		outc.WriteDone()
	}
}

func (c *Symm2D) is2D() bool {
	return c.size[0] == 1
}

func (c *Symm2D) is3D() bool {
	return !c.is2D()
}

func (c *Symm2D) Output() *data.Quant {
	return c.output
}

// TODO: kernel is *Slice
func NewConvolution(input *data.Quant, kernel [3][3][][][]float32) *Symm2D {
	mesh := input.Mesh()
	size := mesh.Size()
	c := new(Symm2D)
	c.size = size
	for i := 0; i < 3; i++ {
		for j := 0; j < 3; j++ {
			if kernel[i][j] != nil {
				c.kern[i][j] = data.SliceFromList([][]float32{data.Contiguous(kernel[i][j])}, mesh)
			}
		}
	}
	c.n = prod(size)
	c.kernSize = data.SizeOf(kernel[0][0])
	c.input = input.NewReader()
	c.output = NewQuant(3, mesh)

	c.init()
	//nimble.Stack(c)

	return c
	// TODO: self-test
}
