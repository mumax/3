package gpu

import (
	"code.google.com/p/mx3/core"
)

// Brute-force O(N²) vector convolution on CPU.
// Used to verify GPU FFT convolution.
// Input better be sparse.
// A nil kernel element is interpreted as all 0s.
// Kernel indices are destination index, source index.
//
// 	(O0)   (K01 K02 K03)   (I0)
// 	(O1) = (K11 K12 K13) * (I1)
// 	(O2)   (K21 K22 K23)   (I2)
func Brute(in, out [3][][][]float32, kern [3][3][][][]float32) {

	size := core.SizeOf(in[0])
	ksize := core.SizeOf(kern[0][0])

	// Zero output first
	for c := 0; c < 3; c++ {
		for x := 0; x < size[0]; x++ {
			for y := 0; y < size[1]; y++ {
				for z := 0; z < size[2]; z++ {
					out[c][x][y][z] = 0
				}
			}
		}
	}

	for sc := 0; sc < 3; sc++ {
		for sx := 0; sx < size[0]; sx++ {
			for sy := 0; sy < size[1]; sy++ {
				for sz := 0; sz < size[2]; sz++ {
					if in[sc][sx][sy][sz] == 0 {
						continue // skip zero source
					}
					for dc := 0; dc < 3; dc++ {
						if kern[dc][sc] == nil {
							continue // skip zero kernel
						}
						for dx := 0; dx < size[0]; dx++ {
							i := Wrap(dx-sx, ksize[0])
							for dy := 0; dy < size[1]; dy++ {
								j := Wrap(dy-sy, ksize[1])
								for dz := 0; dz < size[2]; dz++ {
									k := Wrap(dz-sz, ksize[2])
									out[dc][dx][dy][dz] += in[sc][sx][sy][sz] * kern[dc][sc][i][j][k]
								}
							}
						}
					}
				}
			}
		}
	}
}

// Wraps an index to [0, max] by adding/subtracting a multiple of max.
func Wrap(number, max int) int {
	for number < 0 {
		number += max
	}
	for number >= max {
		number -= max
	}
	return number
}
