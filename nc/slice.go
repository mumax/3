package nc

import ()

// Slice is an alias for []float32,
// implementing some useful interfaces.
type Slice []float32

func NewSlice(length int) Slice {
	return make(Slice, length)
}

func (s Slice) Range(i1, i2 int) []float32 {
	return ([]float32(s))[i1:i2]
}

