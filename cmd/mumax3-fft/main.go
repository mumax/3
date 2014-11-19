package main

import (
	"bufio"
	"flag"
	"fmt"
	"io"
	"math"
	"os"
	"path"
	"strings"

	"github.com/barnex/fftw"
	"github.com/mumax/3/httpfs"
	"github.com/mumax/3/util"
)

var (
	flag_Re     = flag.Bool("re", false, "output real part")
	flag_Im     = flag.Bool("im", false, "output imaginary part")
	flag_Mag    = flag.Bool("mag", false, "output magnitude")
	flag_Ph     = flag.Bool("ph", false, "output phase")
	flag_Pad    = flag.Int("zeropad", 1, "zero-pad input by N times its size")
	flag_Win    = flag.String("window", "boxcar", "apply windowing function")
	flag_Stdout = flag.Bool("stdout", false, "output to stdout instead of file")
	flag_Interp = flag.Bool("interpolate", true, "re-sample intput at equidistant points")
)

func main() {
	// process flags
	flag.Parse()
	// no flags: output magnitude
	if !(*flag_Re || *flag_Im || *flag_Mag || *flag_Ph) {
		*flag_Mag = true
	}

	// process files
	for _, f := range flag.Args() {
		outFname := util.NoExt(f) + "_fft" + path.Ext(f)
		doFile(f, outFname)
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

func doFile(infname, outfname string) {

	// read input file
	in := httpfs.MustOpen(infname)
	defer in.Close()

	header, data := ReadTable(in)

	// Process data
	if *flag_Interp {
		data = interp(data) // interpolate to equidistant times
	}

	// determine frequency step, beware zero padding
	const TIME = 0 // time column
	rows := len(data[TIME])
	deltaT := data[TIME][rows-1] - data[TIME][0]
	deltaT *= float32(*flag_Pad)
	deltaF := 1 / (deltaT)

	window := windows[*flag_Win]
	if window == nil {
		panic(fmt.Sprint("invalid window:", *flag_Win, " options:", windows))
	}

	for c := range data {
		applyWindow(data[c], window)
		data[c] = zeropad(data[c], rows*(*flag_Pad))
	}

	transf := FFT(data[1:]) // FFT all but first (time) column

	// write output file
	var out io.Writer
	if *flag_Stdout {
		out = os.Stdout
	} else {
		o := httpfs.MustCreate(outfname)
		defer o.Close()
		out = o
	}
	writeOutput(out, header, transf, deltaF)
}

func writeOutput(out io.Writer, header []string, transf [][]complex64, deltaF float32) {
	// write header
	Fprint(out, "# f(Hz)")
	for _, h := range header[1:] {
		for _, o := range outputs {
			if !*o.Enabled {
				continue
			}
			Fprint(out, "\t", o.Name, "_", h)
		}
	}
	Fprint(out, "\n")

	// write data
	for r := range transf[0] {
		// frequency first
		freq := float32(r) * deltaF
		Fprint(out, freq)

		// then FFT data
		for c := range transf {
			v := transf[c][r]
			for _, o := range outputs {
				if !*o.Enabled {
					continue
				}
				Fprint(out, "\t", o.Filter(v))
			}
		}
		Fprint(out, "\n")
	}
}

func interp(data [][]float32) [][]float32 {

	interp := make([][]float32, len(data))
	for i := range interp {
		interp[i] = make([]float32, len(data[i]))
	}

	const TIME = 0 // time column
	rows := len(data[TIME])
	deltaT := data[TIME][rows-1] - data[TIME][0]

	time := data[TIME]
	si := 0                             // source index
	for di := 0; di < len(time); di++ { // dst index
		want := float32(di) * deltaT / float32(len(time)) // wanted time
		for si < len(time)-1 && !(time[si] <= want && time[si+1] > want && time[si] != time[si+1]) {
			si++
		}

		x := (want - time[si]) / (time[si+1] - time[si])
		if x < 0 || x > 1 {
			panic(fmt.Sprint("x=", x))
		}

		for c := range interp {
			interp[c][di] = (1-x)*data[c][si] + x*data[c][si+1]
		}
	}
	return interp
}

func zeropad(data []float32, length int) []float32 {
	if len(data) == length {
		return data
	}
	padded := make([]float32, length)
	copy(padded, data)
	return padded
}

func FFT(data [][]float32) [][]complex64 {
	cols := len(data)
	transf := make([][]complex64, cols)

	for c := range transf {
		rows := len(data[c])
		transf[c] = make([]complex64, rows/2+1)
		plan := fftw.PlanR2C([]int{rows}, data[c], transf[c], fftw.ESTIMATE)
		plan.Execute()

		// normalize FFT
		norm := float32(math.Sqrt(float64(len(data[c]))))
		for i := range transf[c] {
			transf[c][i] /= complex64(complex(norm, 0))
		}
	}

	return transf
}

func ReadTable(in_ io.Reader) (header []string, data [][]float32) {
	in := bufio.NewReader(in_)

	header = readHeader(in)
	cols := len(header)
	data = readData(in, cols)

	return
}

func readData(in *bufio.Reader, cols int) [][]float32 {
	data := make([][]float32, cols)
	var v float32
	var err error
	c := 0 // current column
	_, err = fmt.Fscan(in, &v)
	for err == nil {
		data[c] = append(data[c], v)
		c = (c + 1) % cols
		_, err = fmt.Fscan(in, &v)
	}
	if err != io.EOF {
		panic(err)
	}
	if c != 0 {
		panic("truncated data")
	}
	return data
}

func readHeader(in *bufio.Reader) []string {
	hdrBytes, _, err2 := in.ReadLine()
	check(err2)
	hdr := string(hdrBytes)
	if hdr[0] != '#' {
		panic(fmt.Sprint("invalid table header: ", hdr))
	}
	hdr = hdr[2:]
	quants := strings.Split(hdr, "\t")
	return quants
}

func check(err error) {
	if err != nil {
		panic(err)
	}
}

func Fprint(out io.Writer, x ...interface{}) {
	_, err := fmt.Fprint(out, x...)
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
