package mag

// Utilities for setting magnetic configurations.

import (
	"nimble-cube/core"
)

// Initializes array uniformly to the vector value.
func Uniform(array [3][][][]float32, value Vector) {
	for c, comp := range array {
		a := core.Contiguous(comp)
		for i := range a {
			a[i] = value[c]
		}
	}
}

// Sets value at index i,j,k to f(i,j,k).
func SetAll(array [3][][][]float32, f func(i, j, k int) Vector) {
	n := core.SizeOf(array[0])
	i2, j2, k2 := n[0], n[1], n[2]
	SetRegion(array, 0, i2, 0, j2, 0, k2, f)
}

// Sets the region between (i1, j1, k1), (i2, j2, k2) to f(i,j,k).
func SetRegion(array [3][][][]float32, i1, i2, j1, j2, k1, k2 int, f func(i, j, k int) Vector) {
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
