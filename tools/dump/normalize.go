package main

import (
	"math"
	"code.google.com/p/nimble-cube/dump"
)

// normalize vector data to unit length
func normalize(f *dump.Frame) {
	a := f.Vectors()
	for i := range a[0] {

		for j := range a[0][i] {

			for k := range a[0][i][j] {
				x, y, z := a[0][i][j][k], a[1][i][j][k], a[2][i][j][k]
				norm := math.Sqrt(float64(x*x + y*y + z*z))
				invnorm := float32(1)
				if norm != 0 {
					invnorm = float32(1 / norm)
				}
				a[0][i][j][k] *= invnorm
				a[1][i][j][k] *= invnorm
				a[2][i][j][k] *= invnorm

			}
		}
	}
}
