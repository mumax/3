package xc

// Brute-force convolution.

import (
	"github.com/barnex/cuda4/safe"
)

// Brute-force O(N²) vector convolution. 
// Used to verify FFT convolution.
// Input better be sparse.
func BruteConvolution(in_ [3][]float32, kern_ [3][3][]float32, size [3]int) (out_ [3][]float32) {

	// setup 3D arrays
	var in, out [3][][][]float32
	for i := 0; i < 3; i++ {
		in[i] = safe.Reshape3DFloat32(in_[i], size[0], size[1], size[2])
		out_[i] = make([]float32, len(in_[i]))
		out[i] = safe.Reshape3DFloat32(out_[i], size[0], size[1], size[2])
	}
	ksize := PadSize(size)
	var kern [3][3][][][]float32
	for s := 0; s < 3; s++ {
		for d := 0; d < 3; d++ {
			if kern_[s][d] != nil {
				kern[s][d] = safe.Reshape3DFloat32(kern_[s][d], ksize[0], ksize[1], ksize[2])
			}
		}
	}

	for sx := 0; sx < size[0]; sx++ {
		for sy := 0; sy < size[1]; sy++ {
			for sz := 0; sz < size[2]; sz++ {
				for sc := 0; sc < 3; sc++ {
					if in[sc][sx][sy][sz] == 0 {
						continue
					}

					for dx := 0; dx < size[0]; dx++ {
						for dy := 0; dy < size[1]; dy++ {
							for dz := 0; dz < size[2]; dz++ {
								for dc := 0; dc < 3; dc++ {

									out[dc][dx][dy][dz] += in[sc][sx][sy][sz] *
										kern[sc][dc][Wrap(dx-sx, ksize[0])][Wrap(dy-sy, ksize[1])][Wrap(dz-sz, ksize[2])]

								}
							}
						}
					}
				}
			}
		}
	}

	return
}
