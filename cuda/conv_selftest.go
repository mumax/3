package cuda

import (
	"code.google.com/p/mx3/data"
	"log"
	"math/rand"
)

func (c *Symm2D) selfTest() {
	backup := c.input.UnsafeData().HostCopy()
	input := data.NewSlice(3, c.input.Mesh())
	initConvTestInput(input.Vectors())
	data.Copy(c.input.UnsafeData(), input)
	c.Exec()
	output := c.output.UnsafeData().HostCopy()
	data.Copy(c.input.UnsafeData(), backup) // restore input
	backup = nil

	brute := data.NewSlice(3, c.input.Mesh())
	Brute(input.Vectors(), brute.Vectors(), c.kern)

	a, b := output.Host(), brute.Host()
	err := float32(0)
	for c := range a {
		for i := range a[c] {
			if abs(a[c][i]-b[c][i]) > err {
				err = abs(a[c][i] - b[c][i])
			}
		}
	}
	if err > CONV_TOLERANCE {
		log.Fatal("convolution self-test error: ", err)
	} else {
		log.Println("convolution self-test error:", err)
	}
}

const CONV_TOLERANCE = 1e-3

func abs(x float32) float32 {
	if x < 0 {
		return -x
	}
	return x
}

// Brute-force O(N²) vector convolution on CPU.
// Used to verify GPU FFT convolution.
// Input better be sparse.
// A nil kernel element is interpreted as all 0s.
// Kernel indices are destination index, source index.
//
// 	(O0)   (K01 K02 K03)   (I0)
// 	(O1) = (K11 K12 K13) * (I1)
// 	(O2)   (K21 K22 K23)   (I2)
func Brute(in, out [3][][][]float32, kernel [3][3]*data.Slice) {

	var kern [3][3][][][]float32
	for i := range kern {
		for j := range kern[i] {
			if kernel[i][j] != nil {
				kern[i][j] = kernel[i][j].Scalars()
			}
		}
	}

	size := data.SizeOf(in[0])
	ksize := data.SizeOf(kern[0][0])
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

// random number between -1 and 1.
func rnd() float32 {
	return 1 - 2*rand.Float32()
}

// generate sparse input data for testing the convolution.
func initConvTestInput(input [3][][][]float32) {
	size := data.SizeOf(input[0])
	N0, N1, N2 := size[0], size[1], size[2]
	is := [...]int{0, N0 / 5, N0 / 2, N0 - 1}
	js := [...]int{0, N1 / 7, N1 / 2, N1 - 1}
	ks := [...]int{0, N2 / 11, N2 / 2, N2 - 1}
	for c := range input {
		for _, i := range is {
			for _, j := range js {
				for _, k := range ks {
					input[c][i][j][k] = rnd()
				}
			}
		}
	}
}
