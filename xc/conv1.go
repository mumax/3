package xc

import (
	"fmt"
	"github.com/barnex/cuda4/cu"
	"github.com/barnex/cuda4/safe"
	"github.com/barnex/fmath"
	"nimble-cube/core"
)

type Conv1 struct {
	size          [3]int             // 3D size of the input/output data
	n             int                // product of size
	input, output [3][]float32       // input/output arrays, 3 component vectors
	realBuf       [3]safe.Float32s   // gpu buffer for real-space, unpadded input/output data
	fftRBuf       [3]safe.Float32s   // Real ("input") buffers for FFT, shares underlying storage with fftCBuf
	fftCBuf       [3]safe.Complex64s // Complex ("output") for FFT, shares underlying storage with fftRBuf
	fwPlan        [3]safe.FFT3DR2CPlan
	bwPlan        [3]safe.FFT3DC2RPlan
	fftKern       [3][3][]float32     // FFT kernel on host
	gpuKern       [3][3]safe.Float32s // FFT kernel on device: TODO: xfer if needed
	push          chan int            // signals input is ready up to the upper limit sent here
	pull          chan int            // signals output is ready up to upper limit sent here
	inframe       chan int            // signals one full input frame has been processed
	inAvailable   int                 // upper bound to where the input array is ready
	inSent        [3]int              // upper bounds to where the input has been sent to device, per component
	outAvailable  int                 // portion of output that is ready
	cpyStr        cu.Stream           // stream for copies
	fftStr        [3]cu.Stream        // streams for ffts of each component
	noKernMul     bool                // disable kernel multiplication, used for self-test
}

// _______________________________________________ run

func (c *Conv1) run() {
	core.Debug("run")

	core.LockCudaThread()
	c.init() // continue initialization here, inside locked CUDA thread

	for {
		c.uploadInputFrameAndFFTAsync()
		c.syncFFTs()
		c.kernMul()
		c.bwFFTAndDownloadAsync()
		c.syncFFTs()
		c.downloadOutputFrame()
	}

}

// _______________________________________________ download output

// Update the output array at least to upto.
// Blocks if needed.
func (c *Conv1) Pull(upto int) {
	if upto > c.n {
		panic(fmt.Errorf("xc.Conv1: Pull: upto out of bounds: %v", upto))
	}

	for upto > c.outAvailable {
		c.outAvailable = <-c.pull
	}
	if upto == c.n { // finished a frame, start over
		c.outAvailable = 0
	}
}

func (c *Conv1) downloadOutputFrame() {
	c.pull <- c.n
}

// _________________________________________________ convolution

func (c *Conv1) syncFFTs() {
	for _, s := range c.fftStr {
		s.Synchronize()
	}
}

func (c *Conv1) bwFFTAndDownloadAsync() {

	padded := PadSize(c.size)
	offset := [3]int{0, 0, 0}
	for i := 0; i < 3; i++ {
		c.bwPlan[i].Exec(c.fftCBuf[i], c.fftRBuf[i]) // uses stream c.fftStr[i]
		copyPad(c.realBuf[i], c.fftRBuf[i], c.size, padded, offset, c.fftStr[i])
		c.realBuf[i].CopyDtoHAsync(c.output[i], c.fftStr[i])
	}

	//TODO: slice only the last one in blocks
	// have the size depend on pull(N) (limited to maxxfer)
}

// Kernel multiplication. 
// FFT's have to be synced first.
func (c *Conv1) kernMul() {
	if c.noKernMul {
		core.Debug("skipping kernMul")
		return
	}

	core.Debug("kernMul")
	kernMul(c.fftCBuf,
		c.gpuKern[0][0], c.gpuKern[1][1], c.gpuKern[2][2],
		c.gpuKern[1][2], c.gpuKern[0][2], c.gpuKern[0][1],
		c.cpyStr)
	c.cpyStr.Synchronize()
}

// Copy+zeropad input buffer (realBuf) to FFT buffer (fftRBuf),
// then in-place FFT. Asynchronous.
func (c *Conv1) fwFFTAsyncComp(i int) {
	padded := PadSize(c.size)
	offset := [3]int{0, 0, 0}
	c.fftRBuf[i].MemsetAsync(0, c.fftStr[i]) // copypad does NOT zero remainder.
	copyPad(c.fftRBuf[i], c.realBuf[i], padded, c.size, offset, c.fftStr[i])
	c.fwPlan[i].Exec(c.fftRBuf[i], c.fftCBuf[i])
}

// ________________________________________________ upload input

// Signals input[0:upper] is ready to be uploaded.
// Only blocks if upper == len(input), after which
// input may safely be overwritten by a new frame.
func (c *Conv1) Push(upper int) {
	if upper > c.n {
		panic(fmt.Errorf("xc.Conv1: Push: upper out of bounds: %v", upper))
	}
	c.push <- upper
	if upper == c.n {
		// wait until input is uploaded before allowing overwrite
		<-c.inframe
	}
}

// Upload one full input array to the GPU.
// Start asynchronous FFT's on each component as soon as possible.
// Wait for them by c.fftStr.Synchronize()
func (c *Conv1) uploadInputFrameAndFFTAsync() {
	ready := false
	for !ready {

		c.updInAvailableWait()
		for c.haveInput() {
			c.sendSomeInput()
			c.updInAvailbleNoWait()
		}
		ready = c.inSent[0] == c.n &&
			c.inSent[1] == c.n &&
			c.inSent[2] == c.n
	}
	c.inframe <- 1 // allow overwrite of input frame
	c.inSent = [3]int{0, 0, 0}
}

// when X,Y,Z components compete for bandwith, use
// this xfer block size.
var maxXfer = 1024 * 1024 / 4 // 1MB todo: optimize.

// Send part of the available input to the GPU.
// preferentially send X component if possible, then Y, then Z.
// Transfer sizes are limited to avoid sending a huge Z block
// when not all X's have been transferred yet, e.g.
func (c *Conv1) sendSomeInput() {
	for i, sent := range c.inSent {
		if sent < c.inAvailable {
			upper := c.inAvailable

			// limit xfer size when sensible
			if i > 0 { // X is never limited: should be first
				if c.inSent[i-1] != c.n { // limit only if previous component not yet ready
					if upper-sent > maxXfer {
						upper = sent + maxXfer
					}
				}
			}

			c.realBuf[i].Slice(sent, upper).CopyHtoDAsync(c.input[i][sent:upper], c.fftStr[i])
			c.inSent[i] = upper
			if c.inSent[i] == c.n { // component ready
				c.fwFFTAsyncComp(i) // start FFT'ing it, uses c.fftStr[i]
			}
			return // stop here so new input can first flow in
		}
	}
}

// Is new  input available? 
// I.e.: input that is ready but has not yet been sent.
func (c *Conv1) haveInput() bool {
	for _, sent := range c.inSent {
		if sent < c.inAvailable {
			return true
		}
	}
	return false
}

// Update c.inAvailable, the upper bound of ready input data.
// Blocks until some new input becomes available due to a Push().
func (c *Conv1) updInAvailableWait() {
	c.inAvailable = <-c.push
	c.updInAvailbleNoWait()
}

// Update c.inAvailable, the upper bound of ready input data.
// Does not block. If no new input is available, nothing is updated.
func (c *Conv1) updInAvailbleNoWait() {
	for havemore := true; havemore; {
		select {
		case c.inAvailable = <-c.push: // splice input blocks together
		default:
			havemore = false
		}
	}
}

// _______________________________________________________  init

func (c *Conv1) init() {
	core.Debug("init")

	c.initPageLock()
	c.initFFTKern()
	c.initBuffers() // alloc after kernel, when memory has been freed.
	c.cpyStr = cu.StreamCreate()
}

func (c *Conv1) initPageLock() {
	for i := 0; i < 3; i++ {
		core.MemHostRegister(c.input[i])
		core.MemHostRegister(c.output[i])
	}
}

func (c *Conv1) initBuffers() {
	for i := 0; i < 3; i++ {
		c.realBuf[i] = safe.MakeFloat32s(prod(c.size))
		c.fftCBuf[i] = safe.MakeComplex64s(c.fwPlan[i].OutputLen())
		c.fftRBuf[i] = c.fftCBuf[i].Float().Slice(0, c.fwPlan[i].InputLen())
	}
}

func (c *Conv1) initFFTKern() {
	padded := PadSize(c.size)
	ffted := FFTR2COutputSizeFloats(padded)
	realsize := ffted
	realsize[2] /= 2

	acc := 4
	kern := magKernel(padded, core.CellSize(), core.Periodic(), acc)

	for i := range c.fwPlan {
		c.fftStr[i] = cu.StreamCreate()
		c.fwPlan[i] = safe.FFT3DR2C(padded[0], padded[1], padded[2])
		c.fwPlan[i].SetStream(c.fftStr[i])
		c.bwPlan[i] = safe.FFT3DC2R(padded[0], padded[1], padded[2])
		c.bwPlan[i].SetStream(c.fftStr[i])
	}
	fwPlan := c.fwPlan[0] // could use any

	output := safe.MakeComplex64s(fwPlan.OutputLen())
	defer output.Free()
	input := output.Float().Slice(0, fwPlan.InputLen())

	for i := 0; i < 3; i++ {
		for j := i; j < 3; j++ {
			input.CopyHtoD(kern[i][j])
			fwPlan.Exec(input, output)
			fwPlan.Stream().Synchronize() // !!
			c.fftKern[i][j] = make([]float32, prod(realsize))
			scaleRealParts(c.fftKern[i][j], output.Float(), 1/float32(fwPlan.InputLen()))

			if core.DEBUG {
				core.Debug("~kern:", i, j, ":", core.Format(safe.Reshape3DFloat32(c.fftKern[i][j], realsize[0], realsize[1], realsize[2])))
			}

			// TODO: partially if low on mem.
			c.gpuKern[i][j] = safe.MakeFloat32s(len(c.fftKern[i][j]))
			c.gpuKern[i][j].CopyHtoD(c.fftKern[i][j])
		}
	}
}

// Extract real parts, copy them from src to dst.
// In the meanwhile, check if imaginary parts are nearly zero
// and scale the kernel to compensate for unnormalized FFTs.
func scaleRealParts(dstList []float32, src safe.Float32s, scale float32) {
	srcList := src.Host()

	// Normally, the FFT'ed kernel is purely real because of symmetry,
	// so we only store the real parts...
	maximg := float32(0.)
	maxreal := float32(0.)
	for i := 0; i < src.Len()/2; i++ {
		dstList[i] = srcList[2*i] * scale
		if fmath.Abs(srcList[2*i+0]) > maxreal {
			maxreal = fmath.Abs(srcList[2*i+0])
		}
		if fmath.Abs(srcList[2*i+1]) > maximg {
			maximg = fmath.Abs(srcList[2*i+1])
		}
	}
	// ...however, we check that the imaginary parts are nearly zero,
	// just to be sure we did not make a mistake during kernel creation.
	//core.Debug("FFT Kernel max imaginary part=", maximg)
	//core.Debug("FFT Kernel max real part=", maxreal)
	core.Debug("FFT Kernel max imaginary/real part=", maximg/maxreal)
	if maximg/maxreal > 1e-5 { // TODO: is this reasonable?
		panic(fmt.Errorf("xc: FFT Kernel max imaginary/real part=", maximg/maxreal))
	}

}

func prod(size [3]int) int {
	return size[0] * size[1] * size[2]
}

// Zero-padded size.
func PadSize(size [3]int) [3]int {
	padded := [3]int{
		size[0] * 2,
		size[1] * 2,
		size[2] * 2}
	if padded[0] == 2 {
		padded[0] = 1 // no need to pad 1 layer thickness
	}
	return padded
}

func FFTR2COutputSizeFloats(logicSize [3]int) [3]int {
	return [3]int{logicSize[0], logicSize[1], logicSize[2] + 2}
}

func NewConv1(input, output [3][]float32, size [3]int) *Conv1 {
	c := new(Conv1)
	N := prod(size)
	for c := 0; c < 3; c++ {
		if len(output[c]) != N || len(input[c]) != N {
			panic(fmt.Errorf("xc.Conv1.Init: inconsistent sizes"))
		}
	}
	c.size = size
	c.n = prod(size)
	c.input = input
	c.output = output
	core.Assert(core.NumWarp() > 0)
	c.push = make(chan int, core.NumWarp()) // Buffer up to one frame. Less should not deadlock though.
	c.pull = make(chan int, core.NumWarp()) // !! should buffer up to N/maxXfer ??
	c.inframe = make(chan int)
	go c.run()
	return c
}
