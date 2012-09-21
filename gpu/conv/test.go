package conv

import (
	"fmt"
	"github.com/barnex/fmath"
	"math/rand"
	"nimble-cube/core"
)

// Interface of any convolution.
type Conv interface {
	Input() [3][][][]float32     // Input data
	Output() [3][][][]float32    // Output data
	Kernel() [3][3][][][]float32 // Convolution kernel
	Exec()                       // Executes the convolution
}

type Constructor func(size [3]int, kernel [3][3][][][]float32) Conv

// Test if the convolution gives the same result as the brute-force implementation.
func Test(c Conv) {
	if !*core.Flag_verify {
		core.Log("skipping convolution self-test")
		return
	}
	input := c.Input()
	output := c.Output()
	size := core.SizeOf(input[0])

	// overwrite input/output data, if any.
	in := core.Contiguous3(input)
	out := core.Contiguous3(output)
	for i := range in {
		for j := range in[i] {
			in[i][j] = 0
			out[i][j] = 666
		}
	}

	initConvTestInput(input)

	// Reference solution
	bruteOut := core.MakeVectors(size)
	Brute(input, bruteOut, c.Kernel())
	ref := core.Contiguous3(bruteOut)

	// solution under test
	c.Exec()
	c.Exec()
	c.Exec() // it may fail the 2nd time, eg. 

	checkErr(ref, out)

	// cleanly set input/output to zero
	for i := range in {
		for j := range in[i] {
			in[i][j] = 0
			out[i][j] = 0
		}
	}
}

func checkErr(ref, out [3][]float32) {
	// check if error is OK
	var maxerr float32
	for c := range ref {
		for i := range ref[c] {
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

// random number between -1 and 1.
func rnd() float32 {
	return 1 - 2*rand.Float32()
}

// generate sparse input data
func initConvTestInput(input [3][][][]float32) {
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
}
