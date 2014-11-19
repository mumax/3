/*
mumax3-fft performs a Fourier transform on mumax3 table output. E.g.:
 	mumax3-fft table.txt
will create table_fft.txt with per-column FFTs of the data in table.txt.
The first column will contain frequencies in Hz.


Flags


To see all flags, run:
	mumax3-fft -help


Output


By default, the magnitude of the FFT is output. To output magnitude and phase:
 	mumax3-fft -mag -ph table.txt

To outupt real and imaginary part:
 	mumax3-fft -re -im table.txt


Zero padding


To apply zero padding to the input data:
 	mumax3-fft -zeropad 2 table.txt
this will zero-pad the input to 2x its original size, thus increasing the apparent frequency resolution by 2x.


Windowing



The following windowing functions are provided: boxcar (no windowing), hamming, hann, welch:
 	mumax3-fft -window hann table.txt


License

mumax3-fft inherits the GPLv3 from the FFTW bindings at http://github.com/barnex/fftw

*/
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
)

func main() {
	// process flags
	flag.Parse()
	if !(*flag_Re || *flag_Im || *flag_Mag || *flag_Ph) {
		*flag_Mag = true
	}
	//fmt.Fprint(os.Stderr, "outputting")
	//for _, o := range outputs {
	//	if !*o.Enabled {
	//		continue
	//	}
	//	fmt.Fprint(os.Stderr, " ", o.Name)
	//}
	//fmt.Fprintln(os.Stderr)

	// process files
	for _, f := range flag.Args() {
		outFname := util.NoExt(f) + "_fft" + path.Ext(f)
		//fmt.Println(f, "->", outFname)
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
	//data = interp(data) // interpolate to equidistant times

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

//func interp(data [][]float32)[][]float32){
//
//}

type windowFunc func(float32, float32) float32

func applyWindow(data []float32, window windowFunc) {
	N := float32(len(data))
	for i := range data {
		n := float32(i) / N
		data[i] *= window(n, N)
	}
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

func boxcar(n, N float32) float32 {
	return 1
}

func welch(n, N float32) float32 {
	return 1 - sqr((n-(N-1)/2)/((N-1)/2))
}

func hann(n, N float32) float32 {
	return 0.5 * (1 - cos((2*math.Pi*n)/(N-1)))
}

func hamming(n, N float32) float32 {
	const a = 0.54
	const b = 1 - a
	return a - b*cos((2*math.Pi*n)/(N-1))
}

func sqr(x float32) float32 { return x * x }
func cos(x float32) float32 { return float32(math.Cos(float64(x))) }

var windows = map[string]windowFunc{
	"boxcar":  boxcar,
	"hamming": hamming,
	"hann":    hann,
	"welch":   welch,
}
