package nc

import ()

// VectorSlice is a slice of Vectors,
// stored component-by-component (first all x, then all y, ...).
type VectorSlice [VECCOMP]Slice

// Make a VectorSlice with N vector elements.
func MakeVectorSlice(N int) VectorSlice {
	var v VectorSlice
	storage := make([]float32, VECCOMP*N) //←[ make([]float32, VECCOMP * N) escapes to heap]
	for c := 0; c < VECCOMP; c++ {
		v[c] = storage[c*N : (c+1)*N]
	}
	return v
}

// Get the i'th Vector element.
func (v VectorSlice) Get(i int) Vector { //←[ can inline VectorSlice.Get  VectorSlice.Get v does not escape]
	return Vector{v[X][i], v[Y][i], v[Z][i]}
}

// Set the i'th Vector element.
func (v VectorSlice) Set(i int, value Vector) { //←[ can inline VectorSlice.Set  VectorSlice.Set v does not escape]
	v[X][i] = value[X]
	v[Y][i] = value[Y]
	v[Z][i] = value[Z]
}

// Number of vector elements
func (s VectorSlice) N() int { //←[ can inline VectorSlice.N  VectorSlice.N s does not escape]
	return len(s[0])
}
