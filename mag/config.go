package mag

// Utilities for setting magnetic configurations.

import (
	"nimble-cube/nimble"
)

// Returns a function that returns the vector value for all i,j,k.
func Uniform(x, y, z float32) func(i, j, k int) Vector {
	v := Vector{x, y, z}
	return func(i, j, k int) Vector {
		return v
	}
}

// Sets value at index i,j,k to f(i,j,k).
func SetAll(array [3][][][]float32, f func(i, j, k int) Vector) {
	n := nimble.SizeOf(array[0])
	i2, j2, k2 := n[0], n[1], n[2]
	SetRegion(array, 0, 0, 0, i2, j2, k2, f)
}

// Sets the region between (i1, j1, k1), (i2, j2, k2) to f(i,j,k).
func SetRegion(array [3][][][]float32, i1, j1, k1, i2, j2, k2 int, f func(i, j, k int) Vector) {
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
