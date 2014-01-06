package data

// Resample returns a slice of new size N,
// using nearest neighbor interpolation over the input slice.
func Resample(in *Slice, N [3]int) *Slice {
	In := in.Tensors()
	out := NewSlice(in.NComp(), N)
	Out := out.Tensors()
	size1 := sizeOf(In[0])
	size2 := sizeOf(Out[0])
	for c := range Out {
		for i := range Out[c] {
			i1 := (i * size1[0]) / size2[0]
			for j := range Out[c][i] {
				j1 := (j * size1[1]) / size2[1]
				for k := range Out[c][i][j] {
					k1 := (k * size1[2]) / size2[2]
					Out[c][i][j][k] = In[c][i1][j1][k1]
				}
			}
		}
	}
	return out
}

// Returns the size of block, i.e., len(block), len(block[0]), len(block[0][0]).
func sizeOf(block [][][]float32) [3]int {
	return [3]int{len(block), len(block[0]), len(block[0][0])}
}
