package main

import (
	"flag"
	"fmt"
	"io"
	"math"
	"os"
	"path"
	"runtime"
)

var (
	flag_CleanPhase = flag.Bool("cleanph", false, "output phase without 2pi jumps")
	flag_Im         = flag.Bool("im", false, "output imaginary part")
	flag_Interp     = flag.Bool("interpolate", true, "re-sample intput at equidistant points")
	flag_Inv        = flag.Bool("inv", false, "inverse FFT")
	flag_Mag        = flag.Bool("mag", false, "output magnitude")
	flag_NormCol    = flag.Int("divcol", 0, "divide by this column in fourier space (counts from 1)")
	flag_Pad        = flag.Int("zeropad", 1, "zero-pad input by N times its size")
	flag_Ph         = flag.Bool("ph", false, "output phase")
	flag_Re         = flag.Bool("re", false, "output real part")
	flag_Stdout     = flag.Bool("stdout", false, "output to stdout instead of file")
	flag_Win        = flag.String("window", "boxcar", "apply windowing function")
)

func main() {
	runtime.GOMAXPROCS(runtime.NumCPU() * 2)
	// process flags
	flag.Parse()

	if *flag_CleanPhase {
		*flag_Ph = true
	}
	// no flags: output magnitude
	if !(*flag_Re || *flag_Im || *flag_Mag || *flag_Ph) {
		*flag_Mag = true
	}

	if flag.NArg() == 0 {
		errPrintln("no input files")
		return
	}

	if path.Ext(flag.Arg(0)) == ".ovf" {
		mainSpatial()
	} else {
		mainTable()
	}
}

// list of possible outputs
var outputs = []output{
	{flag_Re, "Re", func(c complex64) float32 { return real(c) }},
	{flag_Im, "Im", func(c complex64) float32 { return imag(c) }},
	{flag_Mag, "Mag", mag},
	{flag_Ph, "Ph", phase},
}

type output struct {
	Enabled *bool
	Name    string
	Filter  func(complex64) float32
}

func check(err error) {
	if err != nil {
		panic(err)
	}
}

func Fprint(out io.Writer, a ...interface{}) {
	_, err := fmt.Fprint(out, a...)
	check(err)
}

func mag(c complex64) float32 {
	re := float64(real(c))
	im := float64(imag(c))
	return float32(math.Sqrt(re*re + im*im))
}

func phase(c complex64) float32 {
	re := float64(real(c))
	im := float64(imag(c))
	return float32(math.Atan2(im, re))
}

func errPrintln(a ...interface{}) {
	fmt.Fprintln(os.Stderr, a...)
}
