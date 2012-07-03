package nc

import ()

type GpuConvBox struct {
	M [3]<-chan GpuBlock
	B [][3]chan<- GpuBlock

	FFTKernel <-chan GpuBlock

	fftBuf [3]GpuBlock
}

func NewConvBox() *GpuConvBox {
	box := new(GpuConvBox)
	Register(box)
	return box
}

func FFTOutputSizeFloats(logicSize [3]int) [3]int {
	return [3]int{logicSize[0], logicSize[1], logicSize[2] + 2}
}

func (box *GpuConvBox) Run() {

	for s := 0; s < NumWarp(); s++ {
		k := RecvGpu(box.FFTKernel)
		Debug("k", s, ":", k.Host())
	}

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
