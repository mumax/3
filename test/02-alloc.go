// +build ignore

package main

import (
	"code.google.com/p/mx3/cuda"
	"code.google.com/p/mx3/data"
)

func main() {
	N0, N1, N2 := 2, 16, 32
	c0, c1, c2 := 1e-9, 1e-9, 1e-9
	mesh := data.NewMesh(N0, N1, N2, c0, c1, c2)

	m := cuda.NewQuant(3, mesh)
	m.Data()
}
