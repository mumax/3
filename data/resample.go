package data

import "log"

// Resample returns a slice of new size N,
// using nearest neighbor interpolation over the input slice.
func Resample(in *Slice, N [3]int) *Slice {
	m1 := in.Mesh()
	w := m1.WorldSize()
	n0, n1, n2 := float64(N[0]), float64(N[1]), float64(N[2])
	pbc := m1.PBC()
	m2 := NewMesh(N[0], N[1], N[2], w[0]/n0, w[1]/n1, w[2]/n2, pbc[:]...)
	out := NewSlice(in.NComp(), m2)
	resampleNearest(out.Tensors(), in.Tensors())
	return out
}

func resampleNearest(out, in [][][][]float32) {
	if len(in) != len(out) {
		log.Panicf("illegal argument: len(out)=%v, len(in)=%v", len(out), len(in))
	}
	size1 := sizeOf(in[0])
	size2 := sizeOf(out[0])
	for c := range out {
		for i := range out[c] {
			i1 := (i * size1[0]) / size2[0]
			for j := range out[c][i] {
				j1 := (j * size1[1]) / size2[1]
				for k := range out[c][i][j] {
					k1 := (k * size1[2]) / size2[2]
					out[c][i][j][k] = in[c][i1][j1][k1]
				}
			}
		}
	}
}

// Returns the size of block, i.e., len(block), len(block[0]), len(block[0][0]).
func sizeOf(block [][][]float32) [3]int {
	return [3]int{len(block), len(block[0]), len(block[0][0])}
}
