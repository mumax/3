package engine

import (
	"code.google.com/p/mx3/cuda"
	"code.google.com/p/mx3/data"
	"math"
)

type fftm struct {
	autosave
}

func (f *fftm) init() {
	f.nComp = 3
	f.name = "fftm"
}

func (q *fftm) Download() *data.Slice {
	mesh := &demag_.FFTMesh
	n := mesh.Size()
	s := data.NewSlice(3, mesh)
	scale := float32(1 / math.Sqrt(float64(n[0]*n[1]*n[2])))
	for i := 0; i < 3; i++ {
		dst := s.Comp(i)
		fft := demag_.FFT(M.buffer, i)
		cuda.Mul(fft, fft, scale) // normalize fft
		data.Copy(dst, fft)
	}
	return s
}
