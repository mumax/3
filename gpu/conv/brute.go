package conv

import (
	"nimble-cube/core"
)

// Brute-force O(N²) vector convolution on CPU. 
// Used to verify GPU FFT convolution.
// Input better be sparse.
// A nil kernel element is interpreted as all 0s.
func Brute(in, out [3][][][]float32, kern [3][3][][][]float32) {

	size := core.SizeOf(in[0])
	ksize := core.SizeOf(kern[0][0])

	for sc := 0; sc < 3; sc++ {
		for sx := 0; sx < size[0]; sx++ {
			for sy := 0; sy < size[1]; sy++ {
				for sz := 0; sz < size[2]; sz++ {
					// skip zero source
					if in[sc][sx][sy][sz] == 0 {
						continue
					}

					for dc := 0; dc < 3; dc++ {
						K := kern[sc][dc]
						// skip zero kernel
						if K == nil {
							continue
						}
						for dx := 0; dx < size[0]; dx++ {
							i := core.Wrap(dx-sx, ksize[0])
							for dy := 0; dy < size[1]; dy++ {
								j := core.Wrap(dy-sy, ksize[1])
								for dz := 0; dz < size[2]; dz++ {
									k := core.Wrap(dz-sz, ksize[2])
									out[dc][dx][dy][dz] += in[sc][sx][sy][sz] * K[i][j][k]

								}
							}
						}
					}
				}
			}
		}
	}
}
