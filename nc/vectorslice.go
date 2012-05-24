package nc

import ()

// VectorSlice is a slice of Vectors,
// stored component-by-component (first all x, then all y, ...).
type VectorSlice []float32

// Make a VectorSlice with N vector elements.
func MakeVectorSlice(N int) VectorSlice {
	return make(VectorSlice, VECCOMP*N)
}

// Get a component as a scalar slice.
// Shares the underlying storage.
func (v VectorSlice) VectorComp(c int) Slice {
	N := v.N()
	return Slice([]float32(v)[c*N : (c+1)*N])
}

// Number of vector elements
func (s VectorSlice) N() int {
	return len(s) / VECCOMP
}
