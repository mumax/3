package conv

import (
	"fmt"
	"github.com/barnex/fmath"
	"math/rand"
	"nimble-cube/core"
	//"os"
)

// Interface of any convolution.
type Conv interface {
	Input() [3][][][]float32     // Input data
	Output() [3][][][]float32    // Output data
	Kernel() [3][3][][][]float32 // Convolution kernel
	Exec()                       // Executes the convolution
}

// Test if the convolution gives the same result as the brute-force implementation.
func Test(c Conv) {
	core.Log("testing convolution")
	input := c.Input()
	output := c.Output()

	// overwrite input/output data, if any.
	in := core.Contiguous3(input)
	out := core.Contiguous3(output)
	for i := range in {
		for j := range in[i] {
			in[i][j] = 0
			out[i][j] = 666
		}
	}

	// generate sparse input data
	size := core.SizeOf(input[0])
	N0, N1, N2 := size[0], size[1], size[2]
	is := [...]int{N0 - 1} //	is := [...]int{0, N0 / 5, N0 / 2, N0 - 1}
	js := [...]int{N1 - 1} //	js := [...]int{0, N1 / 7, N1 / 2, N1 - 1}
	ks := [...]int{N2 - 1} //	ks := [...]int{0, N2 / 11, N2 / 2, N2 - 1}
	for c := range input {
		for _, i := range is {
			for _, j := range js {
				for _, k := range ks {
					input[c][i][j][k] = rnd()
				}
			}
		}
	}

	// Reference solution
	bruteOut := core.MakeVectors(size)
	Brute(input, bruteOut, c.Kernel())
	ref := core.Contiguous3(bruteOut)

	// solution under test
	c.Exec()

	// check if error is OK
	{
		var maxerr float32
		for c := range in {
			for i := range in[c] {
				if fmath.Abs(out[c][i]-ref[c][i]) > maxerr {
					maxerr = fmath.Abs(out[c][i] - ref[c][i])
				}
			}
		}
		const tolerance = 1e-5
		if maxerr > tolerance {
			//	core.Fprint(os.Stderr, "expected:\n")
			//	core.Fprintf(os.Stderr, "% 6e", bruteOut)
			//	core.Fprint(os.Stderr, "got:\n")
			//	core.Fprintf(os.Stderr, "% 6e", c.Output())
			panic(fmt.Errorf("convolution self-test failed with error %v", maxerr))
		}
		core.Log("convolution test error:", maxerr)
	}

	// cleanly set input/output to zero
	for i := range in {
		for j := range in[i] {
			in[i][j] = 0
			out[i][j] = 0
		}
	}
}

// random number between -1 and 1.
func rnd() float32 {
	return 1 - 2*rand.Float32()
}
