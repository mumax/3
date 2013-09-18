package engine

import (
	"code.google.com/p/mx3/data"
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
		fft := demagConv().FFT(M.buffer, vol, i, bsat.LUT1(), regions.Gpu()) // fft buffer re-used in conv
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
func (f *fftm) Save()            { Save(f) } // rm

//type fftmPower struct{}
//
//// Power of FFTM, used for display in web interface.
//// Frequencies in y shifted to center at 0/m.
//func (q *fftmPower) Get() (*data.Slice, bool) {
//	fftm, _ := FFTM.Get()
//	n := fftm.Mesh().Size()
//	c := fftm.Mesh().CellSize()
//	m1 := data.NewMesh(n[0], n[1], n[2]/2, c[0], c[1], c[2])
//	n1 := n[1]
//	n12 := n1 / 2
//	m1.Unit = fftm.Mesh().Unit
//	power := data.NewSlice(3, m1)
//	f := fftm.Vectors()
//	p := power.Vectors()
//	for i := range p[0] {
//		for j := range p[0][i] {
//			for k := range p[0][i][j] {
//				p[0][i][j][k] = sqrt(sqr(f[0][i][(j+n12)%n1][2*k]) + sqr(f[0][i][(j+n12)%n1][2*k+1]))
//				p[1][i][j][k] = sqrt(sqr(f[1][i][(j+n12)%n1][2*k]) + sqr(f[1][i][(j+n12)%n1][2*k+1]))
//				p[2][i][j][k] = sqrt(sqr(f[2][i][(j+n12)%n1][2*k]) + sqr(f[2][i][(j+n12)%n1][2*k+1]))
//			}
//		}
//	}
//	return power, false
//}
//
//
//func sqr(x float32) float32  { return x * x }
//func sqrt(x float32) float32 { return float32(math.Sqrt(float64(x))) }
