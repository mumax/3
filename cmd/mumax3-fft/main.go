/*
mumax3-fft performs a Fourier transform on mumax3 table output. E.g.:
 	mumax3-fft table.txt
will create table_fft.txt with per-column FFTs of the data in table.txt.
The first column will contain frequencies in Hz.

Flags

To see all flags, run:
	mumax3-fft -help

By default, the magnitude of the FFT is output. To output magnitude and phase:
 	mumax3-fft -mag -ph table.txt

To outupt real and imaginary part:
 	mumax3-fft -re -im table.txt

*/
package main

import (
	"bufio"
	"flag"
	"fmt"
	"io"
	"math"
	"path"
	"strings"

	"github.com/barnex/fftw"
	"github.com/mumax/3/httpfs"
	"github.com/mumax/3/util"
)

var (
	flag_Re  = flag.Bool("re", false, "output real part")
	flag_Im  = flag.Bool("im", false, "output imaginary part")
	flag_Mag = flag.Bool("mag", false, "output magnitude")
	flag_Ph  = flag.Bool("ph", false, "output phase")
)

func main() {
	// process flags
	flag.Parse()
	if !(*flag_Re || *flag_Im || *flag_Mag || *flag_Ph) {
		*flag_Mag = true
	}
	fmt.Print("outputting")
	for _, o := range outputs {
		if !*o.Enabled {
			continue
		}
		fmt.Print(" ", o.Name)
	}
	fmt.Println()

	// process files
	for _, f := range flag.Args() {
		outFname := util.NoExt(f) + "_fft" + path.Ext(f)
		fmt.Println(f, "->", outFname)
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

	// do FFT
	transf := FFT(data[1:]) // FFT all but first (time) column

	const TIME = 0 // time column
	rows := len(data[TIME])
	deltaT := data[TIME][rows-1] - data[TIME][0]
	deltaF := 1 / (deltaT)

	// write output file
	out := httpfs.MustCreate(outfname)
	defer out.Close()

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
