package nc

import ()

// Slice is an alias for []float32,
// implementing some useful interfaces.
type Slice []float32

// Make a slice with N scalar elements.
func MakeSlice(N int) Slice {
	return make(Slice, N)
}

// Number of scalar elements
func (s Slice) N() int {
	return len(s)
}

// Set all elements to a
func (s Slice) Memset(a float32) {
	for i := range s {
		s[i] = a
	}
}

func (s Slice) Range(i1, i2 int) []float32 {
	return ([]float32(s))[i1:i2]
}
