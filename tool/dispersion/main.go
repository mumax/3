package main

import (
	"flag"
	"fmt"
	//"github.com/barnex/fftw3/fftwf"
	"nimble-cube/dump"
	"nimble-cube/core"
	"os"
)

func main() {
	flag.Parse()

	N := flag.NArg() // Number of dump files = Number of time points
	say(N, "input files")
	var list []float32

	frames := dump.ReadAllFiles(flag.Args(), dump.CRC_ENABLED)
	var size [3]int
	//t := 0 // time
	totaltime := 0.
	for f := range frames {
		// init sizes from first frame.
		if size[0] == 0 {
			size := f.MeshSize
			size[2] += 2 // padding
			say("Frame contains", size, "pixels (1st component).")
			list = make([]float32, core.Prod(size) * N)
		}
		totaltime = f.Time
	}

}

func say(msg ...interface{}) {
	fmt.Fprintln(os.Stderr, msg...)
}

