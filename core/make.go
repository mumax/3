package core

// Make a 3D block of vectors. Underlying storage is contiguous per component.
func MakeVectors(size [3]int) [3][][][]float32 {
	return [3][][][]float32{MakeFloats(size), MakeFloats(size), MakeFloats(size)}
}

// Make a 3D block of floats. Underlying storage is contiguous.
func MakeFloats(size [3]int) [][][]float32 {
	storage := make([]float32, Prod(size))
	return Reshape(storage, size)
}
