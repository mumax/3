package mag

// Utilities for setting magnetic configurations.

import (
	"code.google.com/p/mx3/core"
)

// puts a magnetic vortex in m
func SetVortex(m [3][][][]float32, circulation, polarization int) {
	cy, cz := len(m[0][0])/2, len(m[0][0][0])/2
	for i := range m[0] {
		for j := range m[0][i] {
			for k := range m[0][0][j] {
				y := j - cy
				x := k - cz
				m[X][i][j][k] = 0
				m[Y][i][j][k] = float32(x * circulation)
				m[Z][i][j][k] = float32(-y * circulation)
			}
		}
		m[Z][i][cy][cz] = 0.
		m[Y][i][cy][cz] = 0.
		m[X][i][cy][cz] = float32(polarization)
	}
}

// Returns a function that returns the vector value for all i,j,k.
func Uniform(x, y, z float32) func(i, j, k int) [3]float32 {
	v := [3]float32{x, y, z}
	return func(i, j, k int) [3]float32 {
		return v
	}
}

// Sets value at index i,j,k to f(i,j,k).
func SetAll(array [3][][][]float32, f func(i, j, k int) [3]float32) {
	n := core.SizeOf(array[0])
	i2, j2, k2 := n[0], n[1], n[2]
	SetRegion(array, 0, 0, 0, i2, j2, k2, f)
}

// Sets the region between (i1, j1, k1), (i2, j2, k2) to f(i,j,k).
func SetRegion(array [3][][][]float32, i1, j1, k1, i2, j2, k2 int, f func(i, j, k int) [3]float32) {
	for i := i1; i < i2; i++ {
		for j := j1; j < j2; j++ {
			for k := k1; k < k2; k++ {
				v := f(i, j, k)
				for c := range array {
					array[c][i][j][k] = v[c]
				}
			}
		}
	}
}
