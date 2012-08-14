package main

import (
	"flag"
	"fmt"
	"github.com/barnex/fftw3"
	"nimble-cube/dump"
)

func main() {
	flag.Parse()
	N := flag.NArg()
	data := make([]complex128, N)

	frames := dump.ReadAllFiles(flag.Args(), dump.CRC_ENABLED)
	t := 0
	for f := range frames {
		data[t] = complex(float64(f.Data[0]), 0)
		t++
	}
	plan := fftw3.PlanDFT1D(data, data, fftw3.FORWARD, fftw3.ESTIMATE)
	plan.Execute()
	for _, x := range data {
		power := real(x)*real(x) + imag(x)*imag(x)
		fmt.Println(power)
	}
}
