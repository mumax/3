package gpu

//
//import (
//	"code.google.com/p/nimble-cube/nimble"
//	"fmt"
//	"github.com/barnex/fmath"
//	"math/rand"
//)
//
//// Interface of any convolution.
//type Conv interface {
//	Input() [3][][][]float32     // Input data
//	Output() [3][][][]float32    // Output data
//	Kernel() [3][3][][][]float32 // Convolution kernel
//	Exec()                       // Executes the convolution
//}
//
//type Constructor func(size [3]int, kernel [3][3][][][]float32) Conv
//
//// Test if the convolution gives the same result as the brute-force implementation.
//func Test(c Conv) {
//	if !*nimble.Flag_verify {
//		nimble.Log("skipping convolution self-test")
//		return
//	}
//	input := c.Input()
//	output := c.Output()
//	size := nimble.SizeOf(input[0])
//
//	// overwrite input/output data, if any.
//	in := nimble.Contiguous3(input)
//	out := nimble.Contiguous3(output)
//	for i := range in {
//		for j := range in[i] {
//			in[i][j] = 0
//			out[i][j] = 666
//		}
//	}
//
//	initConvTestInput(input)
//
//	// Reference solution
//	bruteOut := nimble.MakeVectors(size)
//	Brute(input, bruteOut, c.Kernel())
//	ref := nimble.Contiguous3(bruteOut)
//
//	// solution under test
//	c.Exec()
//	c.Exec()
//	c.Exec() // it may fail the 2nd time, eg.
//
//	checkErr(ref, out)
//
//	// cleanly set input/output to zero
//	for i := range in {
//		for j := range in[i] {
//			in[i][j] = 0
//			out[i][j] = 0
//		}
//	}
//}
//
//func checkErr(ref, out [3][]float32) {
//	// check if error is OK
//	var maxerr float32
//	for c := range ref {
//		for i := range ref[c] {
//			if fmath.Abs(out[c][i]-ref[c][i]) > maxerr {
//				maxerr = fmath.Abs(out[c][i] - ref[c][i])
//			}
//		}
//	}
//	const tolerance = 1e-5
//	if maxerr > tolerance {
//		//	core.Fprint(os.Stderr, "expected:\n")
//		//	core.Fprintf(os.Stderr, "% 6e", bruteOut)
//		//	core.Fprint(os.Stderr, "got:\n")
//		//	core.Fprintf(os.Stderr, "% 6e", c.Output())
//		nimble.Fatal(fmt.Errorf("convolution self-test failed with error %v", maxerr))
//	}
//	nimble.Log("convolution test error:", maxerr, "OK")
//}
//
