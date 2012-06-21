package nc

import (
	"github.com/barnex/cuda4/cu"
	"github.com/barnex/cuda4/cufft"
)

type GpuConvBox struct {
	M      [3]<-chan GpuBlock
	B      [][3]chan<- GpuBlock
	Kernel [3][3]<-chan GpuBlock
	fftBuf [3]GpuBlock
}

func NewGpuConvBox() *GpuConvBox {
	box := new(GpuConvBox)
	Register(box)
	return box
}

func (box *GpuConvBox) Run() {
	LockCudaCtx()

	size := Size()

	// zero-padded size
	padded := [3]int{
		size[0] * 2,
		size[1] * 2,
		size[2] * 2}
	if padded[0] == 2 {
		padded[0] = 1 // no need to pad 1 layer thickness
	}

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

			fftBuf[c].Memset(0) // todo: async

			off0, off1 := 0, 0
			for s := 0; s < NumWarp(); s++ {

				Debug("offset", s, off0, off1, 0)

				m := RecvGpu(box.M[c])
				copyPad(fftBuf[c], m, [3]int{off0, off1, 0}) // todo: async

				// last: update slice offset
				if WarpSize()[0] > 1 {
					off0 += WarpSize()[0]
				} else {
					off1 += WarpSize()[1]
					if off1 >= Size()[1] {
						off1 = 0
						off0++
					}
				}
			}

			Debug("fftbuf:", fftBuf[c].Host())

			fftPlan[c].ExecR2C(fftBuf[c].Pointer(), fftBuf[c].Pointer()) // todo: async?
			fftStream[c].Synchronize()
			//Debug("fftbuf:", fftBuf[c].Host())
		}

		// kernel mul
		for slice := 0; slice < NumWarp(); slice++ {
			//kernmul(fftBuf[0].Slice(slice), fftBuf[1].Slice(slice), fftBuf[2].Slice(slice))
		}
	}
}
