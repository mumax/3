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
	defer g.End()

	g.DrawAxes(0.1, 0.1)
	g.Line(0, 0, 0.5, 0.5)
}
