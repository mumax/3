package util

// Functions for manipulating vector and tensor indices in program and user space

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
	xx = 0
	yy = 1
	zz = 2
	yz = 3
	xz = 4
	xy = 5
	zy = 6
	zx = 7
	yx = 8
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
func SwapIndex(index, dim int) int {
	switch dim {
	default:
		log.Panic("swapindex: invalid dim:", dim)
	case 1:
		return index
	case 3:
		return [3]int{Z, Y, X}[index]
	case 6:
		return [6]int{zz, yy, xx, xy, xz, yz}[index]
	case 9:
		return [9]int{zz, yy, xx, yx, zx, zy, xy, xz, yz}[index]
	}
	panic("unreachable")
}
