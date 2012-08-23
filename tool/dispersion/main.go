package main

import (
	"flag"
	"fmt"
	"github.com/barnex/fftw3"
	"github.com/barnex/reshape"
	//"github.com/barnex/fftw3/fftwf"
	"nimble-cube/dump"
	"os"
)

func main() {
	flag.Parse()

	N := flag.NArg() // Number of dump files = Number of time points
	say(N, "input files")
	var list []float32
	var array [][][][]float32

	frames := dump.ReadAllFiles(flag.Args(), dump.CRC_ENABLED)
	var size [3]int
	t := 0 // time
	//totaltime := 0.
	for f := range frames {
		// init sizes from first frame.
		if size[0] == 0 {
			size := f.MeshSize
			outSize := fftw3.R2COutputSizeFloats(size[:])
			list = make([]float32, prod(outSize)*N)
			array = reshape.R4(list, [4]int{N, size[0], size[1], size[2]})
		}
		//totaltime = f.Time

		in := f.Floats()

		for i := range in {
			for j := range in[i] {
				for k := range in[i][j] {
					array[t][i][j][k] = in[i][j][k]
				}
			}
		}
		t++
	}
}

func say(msg ...interface{}) {
	fmt.Fprintln(os.Stderr, msg...)
}

func prod(n []int)int{
	prod := n[0]
	for i:=range n{
		prod *= n[i]
	}
	return prod
}
