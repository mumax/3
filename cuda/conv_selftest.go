package cuda

// Convolution self-test, performed once at the start of each simulation

import (
	"math/rand"

	"github.com/mumax/3/v3/data"
	"github.com/mumax/3/v3/util"
)

// Compares FFT-accelerated convolution against brute-force on sparse data.
// This is not really needed but very quickly uncovers newly introduced bugs.
func testConvolution(c *DemagConvolution, PBC [3]int, realKern [3][3]*data.Slice) {
	if PBC != [3]int{0, 0, 0} {
		// the brute-force method does not work for pbc.
		util.Log("skipping convolution self-test for PBC")
		return
	}
	util.Log("//convolution self-test...")
	inhost := data.NewSlice(3, c.inputSize)
	initConvTestInput(inhost.Vectors())
	gpu := NewSlice(3, c.inputSize)
	defer gpu.Free()
	data.Copy(gpu, inhost)

	Msat := NewSlice(1, [3]int{1, 1, 256})
	defer Msat.Free()
	Memset(Msat, 1)

	vol := data.NilSlice(1, c.inputSize)
	c.Exec(gpu, gpu, vol, ToMSlice(Msat))

	output := gpu.HostCopy()

	brute := data.NewSlice(3, c.inputSize)
	bruteConv(inhost.Vectors(), brute.Vectors(), realKern)

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
		util.Fatal("convolution self-test tolerance: ", err, " FAIL")
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
		for iz := 0; iz < size[Z]; iz++ {
			for iy := 0; iy < size[Y]; iy++ {
				for ix := 0; ix < size[X]; ix++ {
					out[c][iz][iy][ix] = 0
				}
			}
		}
	}

	for sc := 0; sc < 3; sc++ {

		for sz := 0; sz < size[Z]; sz++ {
			for sy := 0; sy < size[Y]; sy++ {
				for sx := 0; sx < size[X]; sx++ {

					if in[sc][sz][sy][sx] == 0 {
						continue // skip zero source
					}
					for dc := 0; dc < 3; dc++ { // dest component
						if kern[dc][sc] == nil {
							continue // skip zero kernel
						}
						for dz := 0; dz < size[Z]; dz++ {
							k := wrap(dz-sz, ksize[Z])

							for dy := 0; dy < size[Y]; dy++ {
								j := wrap(dy-sy, ksize[Y])

								for dx := 0; dx < size[X]; dx++ {
									i := wrap(dx-sx, ksize[X])

									out[dc][dz][dy][dx] += in[sc][sz][sy][sx] * kern[dc][sc][k][j][i]
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
	for number < 0 {
		number += max
	}
	for number >= max {
		number -= max
	}
	return number
}

// generate sparse input data for testing the convolution.
func initConvTestInput(input [3][][][]float32) {
	rng := rand.New(rand.NewSource(0)) // reproducible tests
	size := sizeOf(input[0])

	Nx, Ny, Nz := size[X], size[Y], size[Z]
	ixs := [...]int{0, Nx / 5, Nx / 2, Nx - 1}
	iys := [...]int{0, Ny / 7, Ny / 2, Ny - 1}
	izs := [...]int{0, Nz / 11, Nz / 2, Nz - 1}

	for c := range input {
		for _, i := range izs {
			for _, j := range iys {
				for _, k := range ixs {
					input[c][i][j][k] = 1 - 2*rng.Float32()
				}
			}
		}
	}
}

// Returns the x, y, z size of block
func sizeOf(block [][][]float32) [3]int {
	return [3]int{len(block[0][0]), len(block[0]), len(block)}
}
