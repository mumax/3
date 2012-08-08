package core

// Brute-force convolution.

// Brute-force O(N²) vector convolution. 
// Used to verify FFT convolution.
// Input better be sparse.
// Kernel assumed symmtetric. 
// If kern[i][j] is nil, use kern[j][i]. 
// If that is nil too, use all 0's.
func BruteSymmetricConvolution(in [3][][][]float32, kern [3][3][][][]float32) [3][][][]float32 {

	size := SizeOf(in[0])
	ksize := SizeOf(kern[0][0])
	out := MakeVectors(size)

	for sc := 0; sc < 3; sc++ {
		for sx := 0; sx < size[0]; sx++ {
			for sy := 0; sy < size[1]; sy++ {
				for sz := 0; sz < size[2]; sz++ {
					// skip zero source
					if in[sc][sx][sy][sz] == 0 {
						continue
					}

					for dc := 0; dc < 3; dc++ {
						k := kern[sc][dc]
						// nil-element means: use symmetric one
						if k == nil {
							k = kern[dc][sc]
						}
						// skip zero kernel
						if k == nil {
							continue
						}
						for dx := 0; dx < size[0]; dx++ {
							for dy := 0; dy < size[1]; dy++ {
								for dz := 0; dz < size[2]; dz++ {
									out[dc][dx][dy][dz] += in[sc][sx][sy][sz] *
										k[Wrap(dx-sx, ksize[0])][Wrap(dy-sy, ksize[1])][Wrap(dz-sz, ksize[2])]

								}
							}
						}
					}
				}
			}
		}
	}
	return out
}
