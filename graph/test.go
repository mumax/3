// +build ignore

package main

import (
	. "."
	"os"
)

func main() {
	out, err := os.OpenFile("test.svg", os.O_CREATE|os.O_TRUNC|os.O_WRONLY, 0666)
	if err != nil {
		panic(err)
	}
	defer out.Close()

	g := New(out, 600, 400)
	g.SetRanges(-1e-9, 5e-9, -1, 2)
	defer g.End()

	g.DrawAxes(1e-9, 0.5)
	g.DrawXLabel("Object-oriented design is the roman numerals of computing.")
	g.Line(0, 0, 1e-9, 0.5)
}
