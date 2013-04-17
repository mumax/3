package engine

// Functions for manipulating vector and tensor indices in program and user space
// Author: Arne Vansteenkiste

import "log"

// Indices for vector components
const (
	X = 0
	Y = 1
	Z = 2
)

// Linear indices for matrix components.
// E.g.: matrix[Y][Z] is stored as list[YZ]
const (
	XX = 0
	YY = 1
	ZZ = 2
	YZ = 3
	XZ = 4
	XY = 5
	ZY = 6
	ZX = 7
	YX = 8
)

// Transforms the index between user and program space, unless it is a scalar:
//	X  <-> Z
//	Y  <-> Y
//	Z  <-> X
//	XX <-> ZZ
//	YY <-> YY
//	ZZ <-> XX
//	YZ <-> XY
//	XZ <-> XZ
//	XY <-> YZ
func swapIndex(index, dim int) int {
	switch dim {
	default:
		log.Panic("swapindex: invalid dim:", dim)
	case 1:
		return index
	case 3:
		return [3]int{Z, Y, X}[index]
	case 6:
		return [6]int{ZZ, YY, XX, XY, XZ, YZ}[index]
	case 9:
		return [9]int{ZZ, YY, XX, YX, ZX, ZY, XY, XZ, YZ}[index]
	}
	panic("unreachable")
}
