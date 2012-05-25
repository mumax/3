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

// Number of scalar elements.
func (s VectorSlice) NFloat() int {
	return len(s) * len(s[0])
}

// Number of vector elements.
func (s VectorSlice) NVector() int {
	return len(s[0])
}

// Returns the contiguous underlying storage.
// Contains first all X component, than Y, than Z.
func (v VectorSlice) Contiguous() Slice {
	return ([]float32)(v[0])[:v.NFloat()]
}

func (v VectorSlice) Range(i1, i2 int) [3][]float32 {
	return [3][]float32{v[X][i1:i2], v[Y][i1:i2], v[Z][i1:i2]}
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
