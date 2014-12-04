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

// main loop for table input
func mainTable() {
	// process files
	for _, f := range flag.Args() {
		outFname := util.NoExt(f) + "_fft" + path.Ext(f)
		processTable(f, outFname)
	}
}

// FFT a single table file
func processTable(infname, outfname string) {

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

	// mumax outputs one row too much
	rows--
	for i := range data {
		data[i] = data[i][:rows]
	}

	window := windows[*flag_Win]
	if window == nil {
		panic(fmt.Sprint("invalid window:", *flag_Win, " options:", windows))
	}

	for c := range data {
		applyWindow(data[c], window)
		data[c] = zeropad(data[c], rows*(*flag_Pad))
	}

	transf := FFT(data[1:]) // FFT all but first (time) column

	if *flag_NormCol == 1 {
		errPrintln("-divcol may start at column 2")
		os.Exit(1)
	}

	if *flag_NormCol > 1 {
		divCol(transf, transf[*flag_NormCol-2])
	}

	output := reduce(transf, deltaF)

	outHdr := makeFFTHeader(header)
	if *flag_CleanPhase {
		for c := range output[1:] {
			if strings.HasPrefix(outHdr[c], "Ph_") {
				cleanPhase(output[c])
			}
		}
	}

	// write output file
	var out io.Writer
	if *flag_Stdout {
		out = os.Stdout
	} else {
		o := httpfs.MustCreate(outfname)
		defer o.Close()
		out = o
	}
	writeTable(out, outHdr, output)
}

// turn original table header into FFT header
func makeFFTHeader(origHeader []string) []string {
	fftHdr := []string{"f (Hz)"}
	for _, h := range origHeader[1:] {
		for _, o := range outputs {
			if !*o.Enabled {
				continue
			}
			fftHdr = append(fftHdr, o.Name+"_"+h)
		}
	}
	return fftHdr
}

func cleanPhase(data []float32) {
	for i := 1; i < len(data); i++ {
		for data[i]-data[i-1] > math.Pi {
			data[i] -= 2 * math.Pi
		}
		for data[i]-data[i-1] < -math.Pi {
			data[i] += 2 * math.Pi
		}
	}
}

func reduce(transf [][]complex64, deltaF float32) [][]float32 {

	// count # outputs per input column
	outPerCol := 0
	for _, o := range outputs {
		if *o.Enabled {
			outPerCol++
		}
	}
	nOut := 1 + outPerCol*len(transf)

	out := make([][]float32, nOut)
	for i := range out {
		out[i] = make([]float32, len(transf[0]))
	}

	for r := range transf[0] {
		// frequency first
		freq := float32(r) * deltaF
		out[0][r] = freq

		i := 0
		for c := range transf {
			v := transf[c][r]
			for _, o := range outputs {
				if !*o.Enabled {
					continue
				}
				out[i+1][r] = o.Filter(v)
				i++
			}
		}
	}
	return out
}

// output FFT data table
func writeTable(out io.Writer, header []string, output [][]float32) {

	WriteHeader(out, header)

	// write data
	for r := range output[0] {
		for c := range output {
			Fprint(out, output[c][r], "\t")
		}
		Fprint(out, "\n")
	}
}

func WriteHeader(out io.Writer, header []string) {
	Fprint(out, "# ", header[0])
	for _, h := range header[1:] {
		Fprint(out, "\t", h)
	}
	Fprint(out, "\n")
}

// interpolate input data to equidistant time points in column 0.
func interp(data [][]float32) [][]float32 {

	interp := make([][]float32, len(data))
	for i := range interp {
		interp[i] = make([]float32, len(data[i]))
	}

	const TIME = 0 // time column
	rows := len(data[TIME])
	deltaT := data[TIME][rows-1] - data[TIME][0]

	time := data[TIME]
	time0 := time[0]                    // start time, not neccesarily 0
	si := 0                             // source index
	for di := 0; di < len(time); di++ { // dst index
		want := time0 + float32(di)*deltaT/float32(len(time)) // wanted time
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

// zero-padd data to fit length
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

	// remove nyquist freq
	for i := range transf {
		transf[i] = transf[i][:len(transf[i])-1]
	}

	return transf
}

func divCol(data [][]complex64, norm []complex64) {
	for i := range data[0] {
		for c := range data {
			n := 1 / norm[i]
			data[c][i] *= n
		}
	}
}

// read data table
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
