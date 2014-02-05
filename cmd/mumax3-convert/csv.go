package main

import (
	"fmt"
	"github.com/mumax/3/data"
	"io"
)

// comma-separated values
func dumpCSV(out io.Writer, f *data.Slice) {
	f2 := ", " + *flag_format
	a := f.Tensors()
	for _, a := range a {
		for _, a := range a {
			for _, a := range a {
				fmt.Fprintf(out, *flag_format, a[0])
				for i := 1; i < len(a); i++ {
					fmt.Fprintf(out, f2, a[i])
				}
				fmt.Fprintln(out)
			}
			fmt.Fprintln(out)
		}
	}
}
