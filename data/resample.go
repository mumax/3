package data

import (
	"github.com/mumax/3/util"
)

// Resample returns a slice of new size N,
// using nearest neighbor interpolation over the input slice.
func Resample(in *Slice, N [3]int) *Slice {
	if in.Size() == N {
		return in // nothing to do
	}
	In := in.Tensors()
	out := NewSlice(in.NComp(), N)
	Out := out.Tensors()
	size1 := SizeOf(In[0])
	size2 := SizeOf(Out[0])
	for c := range Out {
		for i := range Out[c] {
			i1 := (i * size1[Z]) / size2[Z]
			for j := range Out[c][i] {
				j1 := (j * size1[Y]) / size2[Y]
				for k := range Out[c][i][j] {
					k1 := (k * size1[X]) / size2[X]
					Out[c][i][j][k] = In[c][i1][j1][k1]
				}
			}
		}
	}
	return out
}

// Downsample returns a slice of new size N, smaller than in.Size().
// Averaging interpolation over the input slice.
// In is returned untouched if the sizes are equal.
func Downsample(In [][][][]float32, N [3]int) [][][][]float32 {
	if SizeOf(In[0]) == N {
		return In // nothing to do
	}

	nComp := len(In)
	out := NewSlice(nComp, N)
	Out := out.Tensors()

	srcsize := SizeOf(In[0])
	dstsize := SizeOf(Out[0])

	Dx := dstsize[X]
	Dy := dstsize[Y]
	Dz := dstsize[Z]
	Sx := srcsize[X]
	Sy := srcsize[Y]
	Sz := srcsize[Z]
	scalex := Sx / Dx
	scaley := Sy / Dy
	scalez := Sz / Dz
	util.Assert(scalex > 0 && scaley > 0)

	for c := range Out {

		for iz := 0; iz < Dz; iz++ {
			for iy := 0; iy < Dy; iy++ {
				for ix := 0; ix < Dx; ix++ {
					sum, n := 0.0, 0.0

					for I := 0; I < scalez; I++ {
						i2 := iz*scalez + I
						for J := 0; J < scaley; J++ {
							j2 := iy*scaley + J
							for K := 0; K < scalex; K++ {
								k2 := ix*scalex + K

								if i2 < Sz && j2 < Sy && k2 < Sx {
									sum += float64(In[c][i2][j2][k2])
									n++
								}
							}
						}
					}
					Out[c][iz][iy][ix] = float32(sum / n)
				}
			}
		}
	}

	return Out
}

// Returns the 3D size of block
func SizeOf(block [][][]float32) [3]int {
	return [3]int{len(block[0][0]), len(block[0]), len(block)}
}
