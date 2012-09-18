package core

// Utility functions

import "os"

// Open file for writing, panic or error.
func OpenFile(fname string) *os.File {
	f, err := os.OpenFile(fname, os.O_WRONLY|os.O_TRUNC|os.O_CREATE, 0666)
	PanicErr(err)
	return f
}

// Product of elements.
func Prod(size [3]int) int {
	return size[0] * size[1] * size[2]
}

// Returns the size after zero-padding,
// taking into account periodic boundary conditions.
func PadSize(size, periodic [3]int) [3]int {
	for i := range size {
		if periodic[i] == 0 && size[i] > 1 {
			size[i] *= 2
		}
	}
	return size
}

// Wraps an index to [0, max] by adding/subtracting a multiple of max.
func Wrap(number, max int) int {
	for number < 0 {
		number += max
	}
	for number >= max {
		number -= max
	}
	return number
}

// Panics if a != b
func CheckEqualSize(a, b [3]int) {
	if a != b {
		Panic("Size mismatch:", a, "!=", b)
	}
}

func min(x, y int) int {
	if x < y {
		return x
	}
	return y
}

func max(x, y int) int {
	if x > y {
		return x
	}
	return y
}
