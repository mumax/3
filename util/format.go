package util

import (
	"bytes"
	"fmt"
	"io"
	"os"
)

// Produces nicely formatted output for multi-dimensional arrays.
func Println(array ...any) {
	Fprint(os.Stdout, array...)
	fmt.Fprintln(os.Stdout)
}

// Produces nicely formatted output for multi-dimensional arrays.
func Print(array ...any) {
	Fprint(os.Stdout, array...)
}

// Produces nicely formatted output for multi-dimensional arrays.
func Printf(format string, array ...any) {
	Fprintf(os.Stdout, format, array...)
}

// Produces nicely formatted output for multi-dimensional arrays.
func Fprint(out io.Writer, array ...any) {
	Fprintf(out, "%v", array...)
}

func Sprint(array ...any) string {
	var buf bytes.Buffer
	Fprint(&buf, array...)
	return buf.String()
}

// Produces nicely formatted output for multi-dimensional arrays.
func Fprintf(out io.Writer, format string, array ...any) {
	for _, arr := range array {
		switch a := arr.(type) {
		case [][][]float32:
			FprintfFloats(out, format, a)
		case [][][][]float32:
			FprintfTensors(out, format, a)
		case [3][][][]float32:
			FprintfTensors(out, format, a[:])
		case [3][3][][][]float32:
			Fprintf(out, format, a[0][:])
			Fprintf(out, format, a[1][:])
			Fprintf(out, format, a[2][:])
		default:
			fmt.Fprintf(out, format, a)
		}
	}
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

//// Produces nicely formatted output.
//func FprintComplexs(out io.Writer, a [][][]complex64) {
//	for i := range a {
//		for j := range a[i] {
//			for _, v := range a[i][j] {
//				fmt.Fprint(out, v, " ")
//			}
//			fmt.Fprintln(out)
//		}
//		fmt.Fprintln(out)
//	}
//}
