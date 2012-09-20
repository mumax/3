package core

//// Functions for manipulating vector and tensor indices in program and user space
//// Author: Arne Vansteenkiste
//
//import (
//	"fmt"
//)
//
//// Indices for vector components
//const (
//	X = 0
//	Y = 1
//	Z = 2
//)
//
//// Linear indices for matrix components.
//// E.g.: matrix[Y][Z] is stored as list[YZ]
//const (
//	XX = 0
//	YY = 1
//	ZZ = 2
//	YZ = 3
//	XZ = 4
//	XY = 5
//	ZY = 6
//	ZX = 7
//	YX = 8
//)
//
//// Maps the 3x3 indices of a symmetric matrix (K_ij) onto
//// a length 6 array containing the upper triangular part:
//// (Kxx, Kyy, Kzz, Kyz, Kxz, Kxy)
////var SymmTensorIdx [3][3]int = [3][3]int{
////	[3]int{XX, XY, XZ},
////	[3]int{XY, YY, YZ},
////	[3]int{XZ, YZ, ZZ}}
//
//// Maps the 3x3 indices of a matrix (K_ij) onto linear indices
//// (Kxx, Kyy, Kzz, Kyz, Kxz, Kxy, Kzy, Kzx, Kyx)
////var FullTensorIdx [3][3]int = [3][3]int{
////	[3]int{XX, XY, XZ},
////	[3]int{YX, YY, YZ},
////	[3]int{ZX, ZY, ZZ}}
//
//// Maps a linear index onto matrix indices i,j
////func IdxToIJ(idx int) (i, j int) {
////	i = [9]int{X, Y, Z, Y, X, X, Z, Z, Y}[idx]
////	j = [9]int{X, Y, Z, Z, Z, Y, Y, X, X}[idx]
////	return
////}
//
//// Maps string to tensor index
////var TensorIndex map[string]int = map[string]int{"XX": XX, "YY": YY, "ZZ": ZZ, "YZ": YZ, "XZ": XZ, "XY": XY, "ZY": ZY, "ZX": ZX, "YX": YX}
//
//// Maps sting to vector index
////var VectorIndex map[string]int = map[string]int{"X": X, "Y": Y, "Z": Z}
//
//// Maps tensor index to string
////var TensorIndexStr []string = []string{"XX", "YY", "ZZ", "YZ", "XZ", "XY", "ZY", "ZX", "YX"}
//
//// Maps vector index to string
////var VectorIndexStr []string = []string{"X", "Y", "Z"}
//
//// Swaps the X-Z values of the array.
//// This transforms from user to program space and vice-versa.
//func SwapXYZ(array []float64) {
//	Assert(len(array) == 3 || len(array) == 1)
//	if len(array) == 3 {
//		array[X], array[Z] = array[Z], array[X]
//	}
//	return
//}
//
//// Transforms the index between user and program space, unless it is a scalar:
////	X  <-> Z
////	Y  <-> Y
////	Z  <-> X
////	XX <-> ZZ
////	YY <-> YY
////	ZZ <-> XX
////	YZ <-> XY
////	XZ <-> XZ
////	XY <-> YZ 
//func SwapIndex(index, dim int) int {
//	switch dim {
//	default:
//		panic(fmt.Errorf("bug: dim=%v", dim))
//	case 1:
//		return index
//	case 3:
//		return [3]int{Z, Y, X}[index]
//	case 6:
//		return [6]int{ZZ, YY, XX, XY, XZ, YZ}[index]
//	case 9:
//		return [9]int{ZZ, YY, XX, YX, ZX, ZY, XY, XZ, YZ}[index]
//	}
//	return -1 // silence 6g
//}
