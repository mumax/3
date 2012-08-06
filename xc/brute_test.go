package xc

import (
	"fmt"
	"github.com/barnex/cuda4/safe"
	"github.com/barnex/fmath"
	"nimble-cube/core"
	"os"
	"testing"
)

func TestBruteConv(test *testing.T) {
	acc := 4
	cellsize := [3]float64{1e-9, 1e-9, 1e-9}
	N0, N1, N2 := 8, 8, 8
	size := [3]int{N0, N1, N2}
	padded := PadSize(size)
	kern := magKernel(padded, cellsize, [3]int{0, 0, 0}, acc)
	N := prod(size)
	var in_ [3][]float32
	var in [3][][][]float32
	for c := 0; c < 3; c++ {
		in_[c] = make([]float32, N)
		in[c] = safe.Reshape3DFloat32(in_[c], size[0], size[1], size[2])
	}

	in[0][N0/2][N1/2][N2/2] = 1
	in[1][N0/2][N1/2][N2/2] = 0
	in[2][N0/2][N1/2][N2/2] = 0

	out := BruteSymmetricConvolution(in_, kern, size)

	core.Log("Brute-force convolution:")
	var vec [3][][][]float32
	for c := 0; c < 3; c++ {
		vec[c] = safe.Reshape3DFloat32(out[c], size[0], size[1], size[2])
	}

	f, err := os.OpenFile("brute.out", os.O_CREATE|os.O_TRUNC|os.O_WRONLY, 0666)
	defer f.Close()
	core.PanicErr(err)
	for i := range vec[0] {
		for j := range vec[0][i] {
			for k := range vec[0][i][j] {
				x, y, z := vec[0][i][j][k], vec[1][i][j][k], vec[2][i][j][k]
				norm := fmath.Sqrt(x*x + y*y + z*z)
				if norm == 0 {
					norm = 1
				}
				fmt.Fprintln(f, i, j, k, "\t", x/norm, y/norm, z/norm)
			}
		}
	}

}
