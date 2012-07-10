package xc

import (
	"fmt"
	"github.com/barnex/cuda4/cu"
	"github.com/barnex/cuda4/safe"
	"github.com/barnex/fmath"
	"nimble-cube/core"
)

type Conv struct {
	size          [3]int
	n             int
	input, output [3][]float32
	realBuf       [3]safe.Float32s
	fftInBuf      [3]safe.Float32s   // Input buffers for FFT, share underlying storage with fftOutBuf
	fftOutBuf     [3]safe.Complex64s // Output buffers for FFT, share underlying storage with fftInBuf
	fwPlan        [3]safe.FFT3DR2CPlan
	bwPlan        [3]safe.FFT3DC2RPlan
	fftKern       [3][3][]float32
	push, pull    chan int
	inframe       chan int     // signals one full input frame has been processed
	inAvailable   int          // upper bound to where the input array is ready
	inSent        [3]int       // upper bounds to where the input has been sent to device, per component
	cpyStr        cu.Stream    // stream for copies
	fftStr        [3]cu.Stream // streams for ffts of each component
}

// _______________________________________________ run

func (c *Conv) run() {
	core.LockCudaThread()
	core.Debug("xc.Conv.run")

	// continue initialization here, inside locked CUDA thread
	c.init()

	for {
		c.uploadInputFrameAndFFT()
		// wait for fft
		//c.downloadOutputFrame()
	}

}

// _______________________________________________ download output

func (c *Conv) Pull() int {
	return <-c.pull
}

func (c *Conv) downloadOutputFrame() {

}

// _________________________________________________ fft input

func (c *Conv) fwFFTComp(i int) {
	core.Debug("xc.Conv: fw FFT component", i)
	padded := PadSize(c.size)
	offset := [3]int{0, 0, 0}
	copyPad(c.fftInBuf[i], c.realBuf[i], padded, c.size, offset, c.fftStr[i])
	c.fftStr[i].Synchronize() // TODO: remove !!!!!!!!!
	core.Debug("padded", i, ":", core.Format(safe.Reshape3DFloat32(c.fftInBuf[i].Host(), padded[0], padded[1], padded[2])))
	c.fwPlan[i].Exec(c.fftInBuf[i], c.fftOutBuf[i])
	c.fftStr[i].Synchronize() // TODO: remove !!!!!!!!!
	fftd0, fftd1, fftd2 := c.fwPlan[i].OutputSize()
	core.Debug("fftd", i, ":", core.FormatComplex(safe.Reshape3DComplex64(c.fftOutBuf[i].Host(), fftd0, fftd1, fftd2)))
}

// ________________________________________________ upload input

// Signals input[0:upper] is ready to be uploaded.
// Only blocks if upper == len(input), after which
// input may safely be overwritten by a new frame.
func (c *Conv) Push(upper int) {
	if upper > c.n {
		panic(fmt.Errorf("xc.Conv: upper out of bounds: %v", upper))
	}
	c.push <- upper
	if upper == c.n {
		core.Debug("xc.Push: waiting to release input frame")
		<-c.inframe
		core.Debug("xc.Push: waiting to release input frame done")
	}
}

// Upload one full input array to the GPU.
// Start asynchronous FFT's on each component as soon as possible.
// Wait for them by c.fftStr.Synchronize()
func (c *Conv) uploadInputFrameAndFFT() {
	ready := false
	for !ready {

		core.Debug("xc.Conv: waiting for input")
		c.updInAvailableWait()
		core.Debug("xc.Conv: done waiting for input")
		for c.haveInput() {
			//core.Debug("xc.Conv: have input")
			c.sendSomeInput()
			c.updInAvailbleNoWait()
		}
		ready = c.inSent[0] == c.n &&
			c.inSent[1] == c.n &&
			c.inSent[2] == c.n
	}
	core.Debug("xc.Conv: finished frame")
	c.inframe <- 1
	core.Debug("xc.Conv: frame released")
	c.inSent = [3]int{0, 0, 0}
}

var maxXfer = 16 // TODO: increase.

// Send part of the available input to the GPU.
// preferentially send X component if possible, then Y, then Z.
// Transfer sizes are limited to avoid sending a huge Z block
// when not all X's have been transferred yet, e.g.
func (c *Conv) sendSomeInput() {
	for i, sent := range c.inSent {
		if sent < c.inAvailable {
			upper := c.inAvailable
			if upper-sent > maxXfer {
				upper = sent + maxXfer
				//core.Debug("xc.Conv: limiting xfer")
			}
			//core.Debug("xc.Conv: sending comp", i, "elems:", upper-sent)
			c.realBuf[i].Slice(sent, upper).CopyHtoDAsync(c.input[i][sent:upper], c.cpyStr)
			c.cpyStr.Synchronize()
			c.inSent[i] = upper
			if c.inSent[i] == c.n { // component ready
				c.fwFFTComp(i) // start FFT'ing it
			}
			return // stop here so new input can first flow in
		}
	}
}

// Is new  input available? 
// I.e.: input that is ready but has not yet been sent.
func (c *Conv) haveInput() bool {
	for _, sent := range c.inSent {
		if sent < c.inAvailable {
			return true
		}
	}
	return false
}

// Update c.inAvailable, the upper bound of ready input data.
// Blocks until some new input becomes available due to a Push().
func (c *Conv) updInAvailableWait() {
	c.inAvailable = <-c.push
	c.updInAvailbleNoWait()
}

// Update c.inAvailable, the upper bound of ready input data.
// Does not block. If no new input is available, nothing is updated.
func (c *Conv) updInAvailbleNoWait() {
	for havemore := true; havemore; {
		select {
		case c.inAvailable = <-c.push:
			//core.Debug("xc.Conv: splicing :-)")
		default:
			havemore = false
		}
	}
}

// _______________________________________________________  init

func (c *Conv) init() {
	core.Debug("xc.Conv.init")

	c.initPageLock()
	c.initFFTKern()
	c.initBuffers() // alloc after kernel, when memory has been freed.
	c.cpyStr = cu.StreamCreate()
}

func (c *Conv) initPageLock() {
	for i := 0; i < 3; i++ {
		core.MemHostRegister(c.input[i])
		core.MemHostRegister(c.output[i])
	}
}

func (c *Conv) initBuffers() {
	// don't leak on 2nd init
	for i := 0; i < 3; i++ {
		c.realBuf[i].Free()
		c.fftOutBuf[i].Free() // also frees fftInBuf, which shares storage
	}

	for i := 0; i < 3; i++ {
		c.realBuf[i] = safe.MakeFloat32s(prod(c.size))
		c.fftOutBuf[i] = safe.MakeComplex64s(c.fwPlan[i].OutputLen())
		c.fftInBuf[i] = c.fftOutBuf[i].Float().Slice(0, c.fwPlan[i].InputLen())
	}
}

func (c *Conv) initFFTKern() {
	padded := PadSize(c.size)
	ffted := FFTR2COutputSizeFloats(padded)
	realsize := ffted
	realsize[2] /= 2

	acc := 4
	kern := magKernel(padded, core.CellSize(), core.Periodic(), acc)
	//core.Debug("kern:", kern)

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
	//defer output.Free()
	input := output.Float().Slice(0, fwPlan.InputLen())

	for i := 0; i < 3; i++ {
		for j := i; j < 3; j++ {
			input.CopyHtoD(kern[i][j])
			fwPlan.Exec(input, output)
			fwPlan.Stream().Synchronize() // !!
			c.fftKern[i][j] = make([]float32, prod(realsize))
			scaleRealParts(c.fftKern[i][j], output.Float(), 1/float32(fwPlan.InputLen()))
			//core.Debug("fftKern", i, j, ":", c.fftKern[i][j])
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

func NewConv(input, output [3][]float32, size [3]int) *Conv {
	c := new(Conv)
	N := prod(size)
	for c := 0; c < 3; c++ {
		if len(output[c]) != N || len(input[c]) != N {
			panic(fmt.Errorf("xc.Conv.Init: inconsistent sizes"))
		}
	}
	c.size = size
	c.n = prod(size)
	c.input = input
	c.output = output
	core.Assert(core.NumWarp() > 0)
	c.push = make(chan int, core.NumWarp()) // Buffer up to one frame. Less should not deadlock though.
	c.pull = make(chan int, core.NumWarp())
	c.inframe = make(chan int)
	go c.run()
	return c
}
