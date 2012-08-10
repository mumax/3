package core

import (
	"fmt"
	"io"
)

func Fprint3Floats(out io.Writer, a [3][][][]float32) {
	FprintTensors(out, a[:])
}

func FprintTensors(out io.Writer, a [][][][]float32) {
	for i := range a {
		FprintFloats(out, a[i])
		fmt.Fprintln(out)
	}
}

func FprintFloats(out io.Writer, a [][][]float32) {
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
