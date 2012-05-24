package nc

import ()

// Slice is an alias for []float32,
// implementing some useful interfaces.
type Slice []float32

// Make a slice with N scalar elements.
func MakeSlice(N int) Slice { //←[ can inline MakeSlice]
	return make(Slice, N) //←[ make(Slice, N) escapes to heap]
}

// Number of scalar elements
func (s Slice) N() int { //←[ can inline Slice.N  Slice.N s does not escape]
	return len(s)
}

// Set all elements to a
func (s Slice) Memset(a float32) { //←[ Slice.Memset s does not escape]
	for i := range s {
		s[i] = a
	}
}

func (s Slice) Range(i1, i2 int) []float32 { //←[ can inline Slice.Range  leaking param: s]
	return ([]float32(s))[i1:i2]
}
