package nc

import ()

// VectorSlice is a slice of Vectors,
// stored component-by-component (first all x, then all y, ...).
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

// Get the i'th Vector element.
//func (v VectorSlice) Get(i int) Vector { 
//	N := v.N() 
//	return Vector{v[X*N+i], v[Y*N+i], v[Z*N+i]}
//}
//
//// Set the i'th Vector element.
//func (v VectorSlice) Set(i int, value Vector) { 
//	N := v.N() 
//	v[X*N+i] = value[X]
//	v[Y*N+i] = value[Y]
//	v[Z*N+i] = value[Z]
//}

// Number of vector elements
func (s VectorSlice) N() int {
	return len(s[0])
}
