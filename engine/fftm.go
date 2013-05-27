package engine

import (
	"code.google.com/p/mx3/cuda"
	"code.google.com/p/mx3/data"
	"math"
)

// FFT of m
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

type fftmPower struct{}

// Power of FFTM, used for display in web interface
func (q *fftmPower) Download() *data.Slice {
	fftm := FFTM.Download()
	n := fftm.Mesh().Size()
	c := fftm.Mesh().CellSize()
	m1 := data.NewMesh(n[0], n[1], n[2]/2, c[0], c[1], c[2])
	m1.Unit = fftm.Mesh().Unit
	power := data.NewSlice(3, m1)
	f := fftm.Vectors()
	p := power.Vectors()
	for i := range p[0] {
		for j := range p[0][i] {
			for k := range p[0][i][j] {
				p[0][i][j][k] = sqrt(sqr(f[0][i][j][2*k]) + sqr(f[0][i][j][2*k+1]))
				p[1][i][j][k] = sqrt(sqr(f[1][i][j][2*k]) + sqr(f[1][i][j][2*k+1]))
				p[2][i][j][k] = sqrt(sqr(f[2][i][j][2*k]) + sqr(f[2][i][j][2*k+1]))
			}
		}
	}
	return power
}

func sqr(x float32) float32  { return x * x }
func sqrt(x float32) float32 { return float32(math.Sqrt(float64(x))) }
