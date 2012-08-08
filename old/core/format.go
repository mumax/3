package core

import (
	"bytes"
	"fmt"
)

func Format(a [][][]float32) string {
	buf := bytes.NewBufferString("\n")
	for i := range a {
		for j := range a[i] {
			for _, v := range a[i][j] {
				fmt.Fprint(buf, v, " ")
			}
			fmt.Fprintln(buf)
		}
		fmt.Fprintln(buf)
	}
	return buf.String()
}

func FormatComplex(a [][][]complex64) string {
	buf := bytes.NewBufferString("\n")
	for i := range a {
		for j := range a[i] {
			for _, v := range a[i][j] {
				fmt.Fprint(buf, v, " ")
			}
			fmt.Fprintln(buf)
		}
		fmt.Fprintln(buf)
	}
	return buf.String()
}
