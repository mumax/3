package main

import (
	"fmt"
	"io"

	"github.com/mumax/3/v3/data"
)

// comma-separated values
func dumpCSV(f *data.Slice, info data.Meta, out io.Writer) {
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
