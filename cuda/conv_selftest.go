package cuda

import (
	"code.google.com/p/mx3/data"
	"log"
	"math/rand"
)

// compares FFT-accelerated convolution against brute-force on sparse data.
func testConvolution(c *DemagConvolution, mesh *data.Mesh) {
	log.Print("verifying convolution ")
	inhost := data.NewSlice(3, mesh)
	initConvTestInput(inhost.Vectors())
	gpu := NewSlice(3, mesh)
	defer gpu.Free()
	data.Copy(gpu, inhost)

	regions := NewBytes(mesh)
	defer regions.Free()
	Bsat := makeFloats([3]int{1, 1, 256})
	defer Bsat.Free()
	Memset(Bsat, 1)
	BsatLUT := LUTPtr(Bsat.DevPtr(0))

	vol := data.NilSlice(1, mesh)
	c.Exec(gpu, gpu, vol, BsatLUT, regions)

	output := gpu.HostCopy()

	brute := data.NewSlice(3, mesh)
	bruteConv(inhost.Vectors(), brute.Vectors(), c.kern)

	a, b := output.Host(), brute.Host()
	err := float32(0)
	for c := range a {
		for i := range a[c] {
			if fabs(a[c][i]-b[c][i]) > err {
				err = fabs(a[c][i] - b[c][i])
			}
		}
	}
	if err > CONV_TOLERANCE {
		log.Fatal("convolution self-test error: ", err)
	} else {
		log.Println("self-test error:", err)
	}
}

// Maximum tolerable error on demag convolution self-test.
const CONV_TOLERANCE = 1e-6

// Brute-force O(N²) vector convolution on CPU.
// Used to verify GPU FFT convolution.
// Input better be sparse.
// A nil kernel element is interpreted as all 0s.
// Kernel indices are destination index, source index.
// 	(O0)   (K01 K02 K03)   (I0)
// 	(O1) = (K11 K12 K13) * (I1)
// 	(O2)   (K21 K22 K23)   (I2)
func bruteConv(in, out [3][][][]float32, kernel [3][3]*data.Slice) {

	var kern [3][3][][][]float32
	for i := range kern {
		for j := range kern[i] {
			if kernel[i][j] != nil {
				kern[i][j] = kernel[i][j].Scalars()
			}
		}
	}

	size := sizeOf(in[0])
	ksize := sizeOf(kern[0][0])
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
							i := wrap(dx-sx, ksize[0])
							for dy := 0; dy < size[1]; dy++ {
								j := wrap(dy-sy, ksize[1])
								for dz := 0; dz < size[2]; dz++ {
									k := wrap(dz-sz, ksize[2])
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

// Wraps an index to [0, max] (python-like modulus)
func wrap(number, max int) int {
	return ((number % max) + max) % max
}

// generate sparse input data for testing the convolution.
func initConvTestInput(input [3][][][]float32) {
	rng := rand.New(rand.NewSource(0)) // reproducible tests
	size := sizeOf(input[0])
	N0, N1, N2 := size[0], size[1], size[2]
	is := [...]int{0, N0 / 5, N0 / 2, N0 - 1}
	js := [...]int{0, N1 / 7, N1 / 2, N1 - 1}
	ks := [...]int{0, N2 / 11, N2 / 2, N2 - 1}
	for c := range input {
		for _, i := range is {
			for _, j := range js {
				for _, k := range ks {
					input[c][i][j][k] = 1 - 2*rng.Float32()
				}
			}
		}
	}
}

// Returns the size of block, i.e., len(block), len(block[0]), len(block[0][0]).
func sizeOf(block [][][]float32) [3]int {
	return [3]int{len(block), len(block[0]), len(block[0][0])}
}
