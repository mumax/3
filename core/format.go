package core

import (
	"fmt"
	"io"
)

// Produces nicely formatted output for multi-dimensional arrays.
// The format is applied to the individual numbers.
func Fprintf(out io.Writer, format string, array ...interface{}) {
	for _, arr := range array {
		switch a := arr.(type) {
		case [][][]float32:
			FprintfFloats(out, format, a)
		case [][][][]float32:
			FprintfTensors(out, format, a)
		case [3][][][]float32:
			FprintfTensors(out, format, a[:])
		default:
			fmt.Fprintf(out, format, a)
		}
	}
}

// Produces nicely formatted output for multi-dimensional arrays.
func Fprint(out io.Writer, array ...interface{}) {
	Fprintf(out, "%v", array...)
}

// Produces nicely formatted output.
func FprintfTensors(out io.Writer, format string, a [][][][]float32) {
	for i := range a {
		FprintfFloats(out, format, a[i])
		fmt.Fprintln(out)
	}
}

// Produces nicely formatted output.
func FprintfFloats(out io.Writer, format string, a [][][]float32) {
	format += " "
	for i := range a {
		for j := range a[i] {
			for _, v := range a[i][j] {
				fmt.Fprintf(out, format, v)
			}
			fmt.Fprintln(out)
		}
		fmt.Fprintln(out)
	}
}

// Produces nicely formatted output.
func FprintComplexs(out io.Writer, a [][][]complex64) {
	for i := range a {
		for j := range a[i] {
			for _, v := range a[i][j] {
				fmt.Fprint(out, v, " ")
			}
			fmt.Fprintln(out)
		}
		fmt.Fprintln(out)
	}
}
