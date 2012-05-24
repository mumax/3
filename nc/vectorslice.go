package nc

import ()

// VectorSlice is a slice of Vectors,
// stored component-by-component (first all x, than all y, ...).
// VectorSlices should be constructed with MakeVectorSlice(),
// not by casting an exiting [3][]float, to guarantee that
// the underlying storage is contiguous.
type VectorSlice [VECCOMP]Slice

// Make a VectorSlice with N vector elements.
func MakeVectorSlice(N int) VectorSlice {
	var v VectorSlice
	storage := make([]float32, VECCOMP*N)
	for c := 0; c < VECCOMP; c++ {
		v[c] = storage[c*N : (c+1)*N]
	}
	return v
}

// Number of vector elements
func (s VectorSlice) N() int {
	return len(s[0])
}

// Returns the contiguous underlying storage.
// Contains first all X component, than Y, than Z.
func (v VectorSlice) Contiguous() Slice {
	return ([]float32)(v[0])[:VECCOMP*v.N()]
}

// Get the i'th Vector element.
func (v VectorSlice) Get(i int) Vector {
	return Vector{v[X][i], v[Y][i], v[Z][i]}
}

// Set the i'th Vector element.
func (v VectorSlice) Set(i int, value Vector) {
	v[X][i] = value[X]
	v[Y][i] = value[Y]
	v[Z][i] = value[Z]
}
