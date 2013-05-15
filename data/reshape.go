package data

// Array reshaping.

import "fmt"

// Re-interpret a contiguous array as a multi-dimensional array of given size.
// Underlying storage is shared.
func reshape(array []float32, size [3]int) [][][]float32 {
	Nx, Ny, Nz := size[0], size[1], size[2]
	if Nx*Ny*Nz != len(array) {
		panic(fmt.Errorf("reshape: size mismatch: %v*%v*%v != %v", Nx, Ny, Nz, len(array)))
	}
	sliced := make([][][]float32, Nx)
	for i := range sliced {
		sliced[i] = make([][]float32, Ny)
	}
	for i := range sliced {
		for j := range sliced[i] {
			sliced[i][j] = array[(i*Ny+j)*Nz+0 : (i*Ny+j)*Nz+Nz]
		}
	}
	return sliced
}
