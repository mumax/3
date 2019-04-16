package main

import (
	"math"

	"github.com/mumax/3/data"
)

// normalize vector data to given length
func normalize(f *data.Slice, length float64) {
	a := f.Vectors()
	for i := range a[0] {
		for j := range a[0][i] {
			for k := range a[0][i][j] {
				x, y, z := a[0][i][j][k], a[1][i][j][k], a[2][i][j][k]
				norm := math.Sqrt(float64(x*x + y*y + z*z))
				invnorm := float32(1)
				if norm != 0 {
					invnorm = float32(length / norm)
				}
				a[0][i][j][k] *= invnorm
				a[1][i][j][k] *= invnorm
				a[2][i][j][k] *= invnorm

			}
		}
	}
}


func threshold(f *data.Slice, value float32) {
	a := f.Scalars()
	for i := range a {
		for j := range a[i] {
			for k := range a[i][j] {
				if float32(math.Abs(float64(a[i][j][k]))) < value {
					a[i][j][k] =0
				}
			}
		}
	}
}

func normpeak(f *data.Slice) {
	a := f.Vectors()
	maxnorm := 0.
	for i := range a[0] {
		for j := range a[0][i] {
			for k := range a[0][i][j] {

				x, y, z := a[0][i][j][k], a[1][i][j][k], a[2][i][j][k]
				norm := math.Sqrt(float64(x*x + y*y + z*z))
				if norm > maxnorm {
					maxnorm = norm
				}

			}
		}
	}
	scale(f, float32(1/maxnorm))
}

func scale(f *data.Slice, factor float32) {
	a := f.Vectors()
	for i := range a[0] {
		for j := range a[0][i] {
			for k := range a[0][i][j] {
				a[0][i][j][k] *= factor
				a[1][i][j][k] *= factor
				a[2][i][j][k] *= factor

			}
		}
	}
}
