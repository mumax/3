package main

import (
	"flag"
	"fmt"
	"github.com/barnex/fftw3"
	"nimble-cube/dump"
	"os"
)

func main() {
	flag.Parse()

	N := flag.NArg() // Number of dump files = Number of time points
	say(N, "input files")
	maxNum := 512 * 1024 * 1024 // Max numbers to store in memory
	maxPix := maxNum / N
	say("processing maximum", maxPix, "pixels per frame.")

	var data [][]float32

	frames := dump.ReadAllFiles(flag.Args(), dump.CRC_ENABLED)
	size := 0 // of the data
	stride := 1 // skip every stide'th pixel
	t := 0 // time
	totaltime := 0.
	for f := range frames {
		// init sizes from first frame.
		if size == 0 {
			size = f.MeshSize[0] * f.MeshSize[1] * f.MeshSize[2]
			say("Frame contains", size, "pixels (1st component).")
			if size > maxPix {
				stride = (size / maxPix) + 1 // should actually divUp
				say("Skipping every", stride, "th pixel")
				size = maxPix / stride
				say("So still processing", size, "pixels")
			}
			data = make([][]float32, size)
			for i := range data {
				data[i] = make([]float32, N)
			}
		}
		totaltime = f.Time
		

		for i := 0; i < size; i ++ {
			data[i][t] = f.Data[i*stride]
		}
		t++
	}

	buffer := make([]complex128, N)
	power := make([]float64, N/2)

	plan := fftw3.PlanDFT1D(buffer, buffer, fftw3.FORWARD, fftw3.ESTIMATE)

	for _, d := range data {
		for t := range d {
			buffer[t] = complex(float64(d[t]), 0)
		}
		plan.Execute()

		for t := range power {
			x := buffer[t]
			p := real(x)*real(x) + imag(x)*imag(x)
			power[t] += p
		}
	}
	for i,p := range power{
		fmt.Println(float64(i) / totaltime, "\t", p)
	}
}

func say(msg ...interface{}) {
	fmt.Fprintln(os.Stderr, msg...)
}
