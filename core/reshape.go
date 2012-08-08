package core

// Array reshaping.

import "fmt"

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

func SizeOf(block [][][]float32) [3]int {
	return [3]int{len(block), len(block[0]), len(block[0][0])}
}

// Reshape the block to one contiguous list.
// Assumes the block's storage is contiguous,
// like returned by MakeFloats.
func Contiguous(block [][][]float32) []float32 {
	N := Prod(SizeOf(block))
	return block[0][0][:N]
}

//func Reshape3DFloat64(array []float64, Nx, Ny, Nz int) [][][]float64 {
//	if Nx*Ny*Nz != len(array) {
//		panic(fmt.Errorf("reshape: size mismatch: %v*%v*%v != %v", Nx, Ny, Nz, len(array)))
//	}
//	sliced := make([][][]float64, Nx)
//	for i := range sliced {
//		sliced[i] = make([][]float64, Ny)
//	}
//	for i := range sliced {
//		for j := range sliced[i] {
//			sliced[i][j] = array[(i*Ny+j)*Nz+0 : (i*Ny+j)*Nz+Nz]
//		}
//	}
//	return sliced
//}
//
//func Reshape3DComplex64(array []complex64, Nx, Ny, Nz int) [][][]complex64 {
//	if Nx*Ny*Nz != len(array) {
//		panic(fmt.Errorf("reshape: size mismatch: %v*%v*%v != %v", Nx, Ny, Nz, len(array)))
//	}
//	sliced := make([][][]complex64, Nx)
//	for i := range sliced {
//		sliced[i] = make([][]complex64, Ny)
//	}
//	for i := range sliced {
//		for j := range sliced[i] {
//			sliced[i][j] = array[(i*Ny+j)*Nz+0 : (i*Ny+j)*Nz+Nz]
//		}
//	}
//	return sliced
//}
//
//func Reshape3DComplex128(array []complex128, Nx, Ny, Nz int) [][][]complex128 {
//	if Nx*Ny*Nz != len(array) {
//		panic(fmt.Errorf("reshape: size mismatch: %v*%v*%v != %v", Nx, Ny, Nz, len(array)))
//	}
//	sliced := make([][][]complex128, Nx)
//	for i := range sliced {
//		sliced[i] = make([][]complex128, Ny)
//	}
//	for i := range sliced {
//		for j := range sliced[i] {
//			sliced[i][j] = array[(i*Ny+j)*Nz+0 : (i*Ny+j)*Nz+Nz]
//		}
//	}
//	return sliced
//}
