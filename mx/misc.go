package mx

import ()

// Product of numbers in list.
func prod(size []int) int {
	prod := 1
	for _, s := range size {
		prod *= s
	}
	return prod
}
