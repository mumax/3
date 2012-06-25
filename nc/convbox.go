package nc

import (
	"github.com/barnex/cuda4/cu"
	"github.com/barnex/cuda4/cufft"
)

type GpuConvBox struct {
	M      [3]<-chan GpuBlock
	B      [][3]chan<- GpuBlock
	Kii    <-chan GpuBlock
	Kjj    <-chan GpuBlock
	Kkk    <-chan GpuBlock
	Kjk    <-chan GpuBlock
	Kik    <-chan GpuBlock
	Kij    <-chan GpuBlock
	fftBuf [3]GpuBlock
}

func NewConvBox() *GpuConvBox {
	box := new(GpuConvBox)
	Register(box)
	return box
}

func (box *GpuConvBox) Run() {

	size := Size()
	padded := PadSize(size)

	// size of fft'd data
	fftSize := [3]int{
		padded[0],
		padded[1],
		padded[2] + 2}

	// buffer for fft'd data
	box.fftBuf = Make3GpuBlock(fftSize)
	fftBuf := box.fftBuf

	// setup fft plans
	var fftPlan, bwPlan [3]cufft.Handle
	var fftStream [3]cu.Stream
	SetCudaCtx()
	for i := range fftPlan {
		fftPlan[i] = cufft.Plan3d(padded[0], padded[1], padded[2], cufft.R2C)
		bwPlan[i] = cufft.Plan3d(padded[0], padded[1], padded[2], cufft.C2R)
		fftStream[i] = cu.StreamCreate()
		fftPlan[i].SetStream(fftStream[i])
		bwPlan[i].SetStream(fftStream[i])
	}

	// run Convolution, run!
	for {
		// FW all components
		for c := 0; c < 3; c++ {

			// copy + zeropad slice
			fftBuf[c].Memset(0) // todo: async
			for s := 0; s < NumWarp(); s++ {
				offset := sliceOffset(s)
				m := RecvGpu(box.M[c])
				copyPad(fftBuf[c], m, offset) // todo: async
			}
			//Debug("fftbuf:", fftBuf[c].Host())

			// fw fft
			SetCudaCtx()
			fftPlan[c].ExecR2C(fftBuf[c].Pointer(), fftBuf[c].Pointer()) // todo: async?
			fftStream[c].Synchronize()
			//Debug("fftbuf:", fftBuf[c].Host())
		}

		// kernel mul
		for slice := 0; slice < NumWarp(); slice++ {
			kernMul(fftBuf,
				RecvGpu(box.Kii),
				RecvGpu(box.Kjj),
				RecvGpu(box.Kkk),
				RecvGpu(box.Kjk),
				RecvGpu(box.Kik),
				RecvGpu(box.Kij),
				slice) // todo: async
		}
	}
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
