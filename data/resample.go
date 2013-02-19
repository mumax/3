package data

import "log"

func Resample(in *Slice, N0, N1, N2 int) *Slice {
	m1 := in.Mesh()
	w := m1.WorldSize()
	n0, n1, n2 := float64(N0), float64(N1), float64(N2)
	pbc := m1.PBC()
	m2 := NewMesh(N0, N1, N2, w[0]/n0, w[1]/n1, w[2]/n2, pbc[:]...)
	out := NewSlice(in.NComp(), m2)
	resampleNearest(out.Tensors(), in.Tensors())
	return out
}

// ResampleNearest copies in to out using nearest-neighbor interpolation.
// len(in) should be == len(out) (the number of vector components).
// The other dimensions are supposedly different and will be interpolated over.
func resampleNearest(out, in [][][][]float32) {
	if len(in) != len(out) {
		log.Panicf("illegal argument: len(out)=%v, len(in)=%v", len(out), len(in))
	}
	size1 := SizeOf(in[0])
	size2 := SizeOf(out[0])
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
