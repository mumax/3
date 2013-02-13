package util

import "fmt"

// ResampleNearest copies in to out using nearest-neighbor interpolation.
// len(in) should be == len(out) (the number of vector components).
// The other dimensions are supposedly different and will be interpolated over.
func ResampleNearest(out, in [][][][]float32) {
	if len(in) != len(out) {
		panic(fmt.Errorf("illegal argument: len(out)=%v, len(in)=%v", len(out), len(in)))
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
