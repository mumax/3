package nc

import (
	"github.com/barnex/cuda4/safe"
	"github.com/barnex/fmath"
)

type KernelBox struct {
	FFTKernel []chan<- Block

	fftKBlocks []Block // len=numwarp, kernel components packed: xx, yy, zz, ...

	fwPlan safe.FFT3DR2CPlan
	bwPlan safe.FFT3DC2RPlan
}

func NewKernelBox() *KernelBox {
	box := new(KernelBox)
	Register(box)
	return box
}

func (box *KernelBox) Run() {
	box.initKern()
	s := 0
	for {
		Send(box.FFTKernel, box.fftKBlocks[s])
		s++
		s %= NumWarp()
	}
}

func (box *KernelBox) initKern() {

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
	//defer output.Free()
	input := output.Float().Slice(0, fwPlan.InputLen())

	ksize := realsize
	ksize[2] *= 6 // 6 kernel components packed
	fftK := make([]float32, prod(ksize))
	MemHostRegister(fftK)

	blocksize := SliceSize(ksize)
	blocklen := prod(blocksize)

	box.fftKBlocks = make([]Block, NumWarp())
	for s := 0; s < NumWarp(); s++ {
		box.fftKBlocks[s] = AsBlock(fftK[s*blocklen:(s+1)*blocklen], blocksize)
	}

	kind := 0
	for i := 0; i < 3; i++ {
		for j := i; j < 3; j++ {

			k := kern[i][j]
			Debug("input:", input)
			Debug("k.List:", k.List)

			SetCudaCtx()
			input.CopyHtoD(k.List)
			//fwPlan.Exec(input, output)

			for s := 0; s < NumWarp(); s++ {
				scaleRealParts(fftK[s*blocklen+kind*(blocklen/6):s*blocklen+(kind+1)*(blocklen/6)], output.Float().Slice(s*blocklen/6, (s+1)*(blocklen/6)), 1/float32(fwPlan.InputLen()))
			}
			kind++
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
	Debug("FFT Kernel max imaginary part=", maximg)
	Debug("FFT Kernel max real part=", maxreal)
	Debug("FFT Kernel max imaginary/real part=", maximg/maxreal)
	if maximg/maxreal > 1e-5 { // TODO: is this reasonable?
		Log("FFT Kernel max imaginary/real part=", maximg/maxreal)
	}
}

// Infinitely sends the block down chan.
//func SendBlock(Chan []chan<- Block, block Block) {
//	for s := 0; s < NumWarp(); s++ {
//		Send(Chan, block.Slice(s))
//	}
//}
