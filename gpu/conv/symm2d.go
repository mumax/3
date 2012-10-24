package conv

import (
	"github.com/barnex/cuda5/cu"
	"github.com/barnex/cuda5/safe"
	"nimble-cube/core"
	"nimble-cube/gpu"
	"unsafe"
)

type Symm2D struct {
	size        [3]int              // 3D size of the input/output data
	kernSize    [3]int              // Size of kernel and logical FFT size.
	fftKernSize [3]int              // Size of real, FFTed kernel
	n           int                 // product of size
	input       [3]gpu.RChan1       // TODO: fuse with input
	output      [3]gpu.Chan1        // TODO: fuse with output
	fftRBuf     [3]safe.Float32s    // FFT input buf for FFT, shares storage with fftCBuf. 
	fftCBuf     [3]safe.Complex64s  // FFT output buf, shares storage with fftRBuf
	gpuFFTKern  [3][3]safe.Float32s // FFT kernel on device: TODO: xfer if needed
	fwPlan      safe.FFT3DR2CPlan   // Forward FFT (1 component)
	bwPlan      safe.FFT3DC2RPlan   // Backward FFT (1 component)
	stream      cu.Stream           // 
	kern        [3][3][]float32     // Real-space kernel
	kernArr     [3][3][][][]float32 // Real-space kernel
	//fftKern     [3][3][]float32     // FFT kernel on host
}

func (c *Symm2D) init() {
	core.Log("initializing 2D symmetric convolution")
	gpu.LockCudaThread()
	defer gpu.UnlockCudaThread()

	padded := c.kernSize

	{ // init FFT plans
		c.stream = cu.StreamCreate()
		c.fwPlan = safe.FFT3DR2C(padded[0], padded[1], padded[2])
		c.fwPlan.SetStream(c.stream)
		c.bwPlan = safe.FFT3DC2R(padded[0], padded[1], padded[2])
		c.bwPlan.SetStream(c.stream)
	}

	{ // init device buffers
		// 2D re-uses fftBuf[1] as fftBuf[0], 3D needs all 3 fftBufs.
		for i := 1; i < 3; i++ {
			c.fftCBuf[i] = gpu.MakeComplexs(prod(fftR2COutputSizeFloats(c.kernSize)) / 2)
		}
		if c.is3D() {
			c.fftCBuf[0] = gpu.MakeComplexs(prod(fftR2COutputSizeFloats(c.kernSize)) / 2)
		} else {
			c.fftCBuf[0] = c.fftCBuf[1]
		}
		for i := 0; i < 3; i++ {
			c.fftRBuf[i] = c.fftCBuf[i].Float().Slice(0, prod(c.kernSize))
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
	//halfkern[1] = halfkern[1]/2 + 1
	fwPlan := c.fwPlan
	output := safe.MakeComplex64s(fwPlan.OutputLen())
	defer output.Free()
	input := output.Float().Slice(0, fwPlan.InputLen())

	// upper triangular part
	fftKern := make([]float32, prod(halfkern))
	for i := 0; i < 3; i++ {
		for j := i; j < 3; j++ {
			if c.kern[i][j] != nil { // ignore 0's
				input.CopyHtoD(c.kern[i][j])
				fwPlan.Exec(input, output)
				fwPlan.Stream().Synchronize() // !!
				scaleRealParts(fftKern, output.Float().Slice(0, prod(halfkern)*2), 1/float32(fwPlan.InputLen()))
				c.gpuFFTKern[i][j] = safe.MakeFloat32s(len(fftKern))
				c.gpuFFTKern[i][j].CopyHtoD(fftKern)
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
	output := gpu.HostFloats(2 * fwPlan.OutputLen()).Complex()
	defer cu.MemFreeHost(unsafe.Pointer(uintptr(output.Pointer()))) // TODO: is Float32s safe with uintptr?
	input := output.Float().Slice(0, fwPlan.InputLen())

	// upper triangular part
	fftKern := make([]float32, prod(halfkern))
	for i := 0; i < 3; i++ {
		for j := i; j < 3; j++ {
			if c.kern[i][j] != nil { // ignore 0's
				input.CopyHtoD(c.kern[i][j])
				fwPlan.Exec(input, output)
				fwPlan.Stream().Synchronize() // !!
				scaleRealParts(fftKern, output.Float().Slice(0, prod(halfkern)*2), 1/float32(fwPlan.InputLen()))
				c.gpuFFTKern[i][j] = gpu.MakeFloats(len(fftKern))
				c.gpuFFTKern[i][j].CopyHtoD(fftKern)
			}
		}
	}
}

func (c *Symm2D) Run() {
	core.Log("running symmetric 2D convolution")
	gpu.LockCudaThread()

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
	padded := c.kernSize
	offset := [3]int{0, 0, 0}

	//N0, N1, N2 := cc.fftKernSize[1], c.fftKernSize[2]
	for i := 0; i < 3; i++ {
		c.input[i].ReadNext(c.n)
		c.fftRBuf[i].MemsetAsync(0, c.stream)
		copyPad(c.fftRBuf[i], c.input[i].UnsafeData(), padded, c.size, offset, c.stream)
		c.fwPlan.Exec(c.fftRBuf[i], c.fftCBuf[i])
		c.stream.Synchronize()
		c.input[i].ReadDone()
	}

	// kern mul
	kernMulRSymm(c.fftCBuf,
		c.gpuFFTKern[0][0], c.gpuFFTKern[1][1], c.gpuFFTKern[2][2],
		c.gpuFFTKern[1][2], c.gpuFFTKern[0][2], c.gpuFFTKern[0][1],
		c.stream)
	c.stream.Synchronize()

	// BW FFT 
	for i := 0; i < 3; i++ {
		c.output[i].WriteNext(c.n)
		c.bwPlan.Exec(c.fftCBuf[i], c.fftRBuf[i])
		copyPad(c.output[i].UnsafeData(), c.fftRBuf[i], c.size, padded, offset, c.stream)
		c.stream.Synchronize()
		c.output[i].WriteDone()
	}
}

func (c *Symm2D) exec2D() {
	padded := c.kernSize
	offset := [3]int{0, 0, 0}

	N1, N2 := c.fftKernSize[1], c.fftKernSize[2]
	// Convolution is separated into 
	// a 1D convolution for x
	// and a 2D convolution for yz.
	// so only 2 FFT buffers are then needed at the same time.

	// FFT x
	c.input[0].ReadNext(c.n)
	c.fftRBuf[0].MemsetAsync(0, c.stream) // copypad does NOT zero remainder.
	copyPad(c.fftRBuf[0], c.input[0].UnsafeData(), padded, c.size, offset, c.stream)
	c.fwPlan.Exec(c.fftRBuf[0], c.fftCBuf[0])
	//c.stream.Synchronize()
	c.input[0].ReadDone()

	// kern mul X	
	kernMulRSymm2Dx(c.fftCBuf[0], c.gpuFFTKern[0][0], N1, N2, c.stream)
	//c.stream.Synchronize()

	// bw FFT x
	c.output[0].WriteNext(c.n)
	c.bwPlan.Exec(c.fftCBuf[0], c.fftRBuf[0])
	copyPad(c.output[0].UnsafeData(), c.fftRBuf[0], c.size, padded, offset, c.stream)
	c.stream.Synchronize()
	c.output[0].WriteDone()

	// FW FFT yz
	for i := 1; i < 3; i++ {
		c.input[i].ReadNext(c.n)
		c.fftRBuf[i].MemsetAsync(0, c.stream)
		copyPad(c.fftRBuf[i], c.input[i].UnsafeData(), padded, c.size, offset, c.stream)
		c.fwPlan.Exec(c.fftRBuf[i], c.fftCBuf[i])
		c.stream.Synchronize()
		c.input[i].ReadDone()
	}

	// kern mul yz
	kernMulRSymm2Dyz(c.fftCBuf[1], c.fftCBuf[2],
		c.gpuFFTKern[1][1], c.gpuFFTKern[2][2], c.gpuFFTKern[1][2],
		N1, N2, c.stream)
	c.stream.Synchronize()

	// BW FFT yz
	for i := 1; i < 3; i++ {
		c.output[i].WriteNext(c.n)
		c.bwPlan.Exec(c.fftCBuf[i], c.fftRBuf[i])
		copyPad(c.output[i].UnsafeData(), c.fftRBuf[i], c.size, padded, offset, c.stream)
		c.stream.Synchronize()
		c.output[i].WriteDone()
	}
}

func (c *Symm2D) is2D() bool {
	return c.size[0] == 1
}

func (c *Symm2D) is3D() bool {
	return !c.is2D()
}

func NewSymm2D(size [3]int, kernel [3][3][][][]float32, input [3]gpu.RChan1, output [3]gpu.Chan1) *Symm2D {
	c := new(Symm2D)
	c.size = size
	c.kernArr = kernel
	for i := 0; i < 3; i++ {
		for j := 0; j < 3; j++ {
			if kernel[i][j] != nil {
				c.kern[i][j] = core.Contiguous(kernel[i][j])
			}
		}
	}
	c.n = prod(size)
	c.kernSize = core.SizeOf(kernel[0][0])
	c.input = input
	c.output = output

	c.init()

	return c
	// TODO: self-test
}
