package xc

import (
	"fmt"
	"github.com/barnex/cuda4/safe"
	"github.com/barnex/fmath"
	"nimble-cube/core"
)

type Conv struct {
	size          [3]int
	n             int
	input, output [3][]float32
	realBuf       [3]safe.Float32s
	fftBuf        [3]safe.Float32s
	fwPlan        safe.FFT3DR2CPlan
	bwPlan        safe.FFT3DC2RPlan
	fftKern       [3][3][]float32
	push, pull    chan int
	inframe       chan int // signals one full input frame has been processed
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
	c.push = make(chan int, core.NumWarp())
	c.pull = make(chan int)
	c.inframe = make(chan int)
	go c.run()
	return c
}

func (c *Conv) run() {
	core.LockCudaThread()
	core.Debug("xc.Conv.run")

	// continue initialization here, inside locked CUDA thread
	c.init()

	for {
		upper := <-c.push
		for havemore := true; havemore; {
			select {
			case upper = <-c.push:
				core.Debug("have more")
			default:
				havemore = false
			}
		}
		core.Debug("upper:", upper)
	}
}

// Signals input[0:upper] is ready to be uploaded.
// Only blocks if upper == len(input), after which
// input may safely be overwritten.
func (c *Conv) Push(upper int) {
	if upper > c.n {
		panic(fmt.Errorf("xc.Conv: upper out of bounds: %v", upper))
	}
	c.push <- upper
	if upper == c.n {
		core.Debug("xc.Conv: waiting to release input frame")
		<-c.inframe
	}
}

func (c *Conv) Pull() int {
	return <-c.pull
}

func (c *Conv) init() {
	core.Debug("xc.Conv.init")

	c.initPageLock()
	c.initFFTKern()
	c.initBuffers() // alloc after kernel, when memory has been freed.
}

func (c *Conv) initPageLock() {
	for i := 0; i < 3; i++ {
		core.MemHostRegister(c.input[i])
		core.MemHostRegister(c.output[i])
	}
}

func (c *Conv) initBuffers() {
	N := prod(c.size)
	// don't leak on 2nd init
	c.realBuf[0].Free()
	c.fftBuf[0].Free()

	r := safe.MakeFloat32s(3 * N)
	c.realBuf = [3]safe.Float32s{r.Slice(0*N, 1*N), r.Slice(1*N, 2*N), r.Slice(2*N, 3*N)}
}

func (c *Conv) initFFTKern() {
	padded := PadSize(c.size)
	ffted := FFTR2COutputSizeFloats(padded)
	realsize := ffted
	realsize[2] /= 2

	acc := 4
	kern := magKernel(padded, core.CellSize(), core.Periodic(), acc)
	//core.Debug("kern:", kern)

	c.fwPlan = safe.FFT3DR2C(padded[0], padded[1], padded[2])
	c.bwPlan = safe.FFT3DC2R(padded[0], padded[1], padded[2])

	output := safe.MakeComplex64s(c.fwPlan.OutputLen())
	defer output.Free()
	//defer output.Free()
	input := output.Float().Slice(0, c.fwPlan.InputLen())

	for i := 0; i < 3; i++ {
		for j := i; j < 3; j++ {
			input.CopyHtoD(kern[i][j])
			c.fwPlan.Exec(input, output)
			c.fftKern[i][j] = make([]float32, prod(realsize))
			scaleRealParts(c.fftKern[i][j], output.Float(), 1/float32(c.fwPlan.InputLen()))
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
