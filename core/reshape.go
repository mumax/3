package core

// Array reshaping.

import "fmt"

// Re-interpret a contiguous array as a multi-dimensional array of given size.
// Underlying storage is shared.
func Reshape(array []float32, size [3]int) [][][]float32 {
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

// Re-interpret a contiguous array as a multi-dimensional array of given size.
// Underlying storage is shared.
func Reshape4D(list []float32, size []int) [][][][]float32 {
	sliced := make([][][][]float32, size[0])
	for i := range sliced {
		sliced[i] = make([][][]float32, size[1])
	}
	for i := range sliced {
		for j := range sliced[i] {
			sliced[i][j] = make([][]float32, size[2])
		}
	}

	for i := range sliced {
		for j := range sliced[i] {
			for k := range sliced[i][j] {
				sliced[i][j][k] = list[((i*size[1]+j)*size[2]+k)*size[3]+0 : ((i*size[1]+j)*size[2]+k)*size[3]+size[3]]
			}
		}
	}
	return sliced
}

func SizeOf(block [][][]float32) [3]int {
	return [3]int{len(block), len(block[0]), len(block[0][0])}
}

// Reshape the block to one contiguous list.
// Assumes the block's storage is contiguous,
// like returned by MakeFloats or Reshape.
func Contiguous(block [][][]float32) []float32 {
	N := Prod(SizeOf(block))
	return block[0][0][:N]
}

// Reshape the vector block to three contiguous lists.
// Assumes the components's storage are individually contiguous,
// like returned by MakeVectors.
func Contiguous3(v [3][][][]float32)[3][]float32{
	return [3][]float32{Contiguous(v[0]),Contiguous(v[1]),  Contiguous(v[2]) }
}
