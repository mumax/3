package engine

import (
	"github.com/mumax/3/data"
	"math"
)

// TODO: composition: fft(anyquant)

var FFTM fftm // FFT of m

func init() {
	DeclROnly(FFTM.Name(), &FFTM, "FFT of m")
}

// FFT of m
type fftm struct{}

func (q *fftm) Get() (quant *data.Slice, recycle bool) {

	dst := data.NewSlice(3, q.Mesh())
	scale := float32(1 / math.Sqrt(float64(M.Mesh().NCell()))) // logical number of cells

	for i := 0; i < 3; i++ {
		fft := demagConv().FFT(M.buffer, vol, i, Bsat.LUT1(), regions.Gpu()) // fft buffer re-used in conv
		comp := dst.Comp(i)
		data.Copy(comp, fft)
		arr := comp.Host()[0]
		for i := range arr {
			arr[i] *= scale
		}
	}
	return dst, false
}

func (f *fftm) NComp() int       { return 3 }
func (f *fftm) Name() string     { return "mFFT" }
func (f *fftm) Unit() string     { return "" }
func (f *fftm) Mesh() *data.Mesh { return &demagConv().FFTMesh }
