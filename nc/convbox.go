package nc

import (
	"github.com/barnex/cuda4/safe"
	"github.com/barnex/fmath"
)

type GpuConvBox struct {
	M [3]<-chan GpuBlock
	B [][3]chan<- GpuBlock

	Kii <-chan GpuBlock
	Kjj <-chan GpuBlock
	Kkk <-chan GpuBlock
	Kjk <-chan GpuBlock
	Kik <-chan GpuBlock
	Kij <-chan GpuBlock

	fftKern [3][3]Block
	//fftKern[]Block

	fftBuf [3]GpuBlock
	fwPlan safe.FFT3DR2CPlan
	bwPlan safe.FFT3DC2RPlan
}

func NewConvBox() *GpuConvBox {
	box := new(GpuConvBox)
	Register(box)
	return box
}

func (box *GpuConvBox) initKern() {

	size := Size()
	padded := PadSize(size)
	ffted := FFTOutputSizeFloats(padded)
	realsize := ffted
	realsize[2] /= 2

	acc := 4
	Debug("Initializing magnetostatic kernel")
	kern := magKernel(padded, CellSize(), Periodic(), acc)
	Debug("Magnetostatic kernel ready")
	Debug("kern:", kern)

	box.fwPlan = safe.FFT3DR2C(padded[0], padded[1], padded[2])
	box.bwPlan = safe.FFT3DC2R(padded[0], padded[1], padded[2])
	fwPlan := box.fwPlan
	//bwPlan := box.bwPlan

	output := safe.MakeComplex64s(fwPlan.OutputLen())
	defer output.Free()
	input := output.Float().Slice(0, fwPlan.InputLen())

	for i := 0; i < 3; i++ {
		for j := i; j < 3; j++ {
			k := kern[i][j]
			input.CopyHtoD(k.List)
			fwPlan.Exec(input, output)
			//Debug("fft output:", output.Host())
			box.fftKern[i][j] = MakeBlock(realsize)
			scaleRealParts(box.fftKern[i][j], output.Float(), 1/float32(fwPlan.InputLen()))
			//Debug("fftKern", i, j, ":", box.fftKern[i][j].Array)
		}
	}

}

// Extract real parts, copy them from src to dst.
// In the meanwhile, check if imaginary parts are nearly zero
// and scale the kernel to compensate for unnormalized FFTs.
func scaleRealParts(dst Block, src safe.Float32s, scale float32) {
	srcList := src.Host()
	dstList := dst.List

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
	Debug("FFT Kernel max imaginary part=", maximg)
	Debug("FFT Kernel max real part=", maxreal)
	Debug("FFT Kernel max imaginary/real part=", maximg/maxreal)
	if maximg/maxreal > 1e-5 { // TODO: is this reasonable?
		Log("FFT Kernel max imaginary/real part=", maximg/maxreal)
	}
}

func FFTOutputSizeFloats(logicSize [3]int) [3]int {
	return [3]int{logicSize[0], logicSize[1], logicSize[2] + 2}
}

func (box *GpuConvBox) init() {
	box.initKern()
}

func (box *GpuConvBox) Run() {

	box.init()
	//
	//	// run Convolution, run!
	//	for {
	//		// FW all components
	//		for c := 0; c < 3; c++ {
	//
	//			// copy + zeropad slice
	//			fftBuf[c].Memset(0) // todo: async
	//			for s := 0; s < NumWarp(); s++ {
	//				offset := sliceOffset(s)
	//				m := RecvGpu(box.M[c])
	//				copyPad(fftBuf[c], m, offset) // todo: async
	//			}
	//			//Debug("fftbuf:", fftBuf[c].Host())
	//
	//			// fw fft
	//			SetCudaCtx()
	//			fftPlan[c].ExecR2C(fftBuf[c].Pointer(), fftBuf[c].Pointer()) // todo: async?
	//			fftStream[c].Synchronize()
	//			//Debug("fftbuf:", fftBuf[c].Host())
	//		}
	//
	//		// kernel mul
	//		for slice := 0; slice < NumWarp(); slice++ {
	//			kernMul(fftBuf,
	//				RecvGpu(box.Kii),
	//				RecvGpu(box.Kjj),
	//				RecvGpu(box.Kkk),
	//				RecvGpu(box.Kjk),
	//				RecvGpu(box.Kik),
	//				RecvGpu(box.Kij),
	//				slice) // todo: async
	//		}
	//	}
}

// Position of slice block number s in its parent block.
func sliceOffset(s int) [3]int {
	Assert(s < NumWarp())
	size := Size()
	w := WarpSize()
	off0, off1 := 0, 0
	if w[0] > 1 {
		off0 = (s * w[0]) % size[0]
	} else {
		off1 = (s * w[1]) % size[1]
		off0 = ((s * w[1]) / size[1]) % size[0]
	}
	return [3]int{off0, off1, 0}
}
