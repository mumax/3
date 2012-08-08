package core

// Utility functions

func Prod(size [3]int) int {
	return size[0] * size[1] * size[2]
}

// Zero-padding
func PadSize(size, periodic [3]int) [3]int {
	for i := range size {
		if periodic[i] > 0 && size[i] > 1 {
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

func CheckEqualSize(a,b [3]int) {
	for i, s := range a {
		if s != b[i]{
			Panic("Size mismatch:", a, "!=", b)
		}
	}
}
