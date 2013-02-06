// +build ignore

package main

import (
	"code.google.com/p/mx3/mx"
)

func main() {
	N0, N1, N2 := 2, 16, 32
	c0, c1, c2 := 1e-9, 1e-9, 1e-9
	mesh := mx.NewMesh(N0, N1, N2, c0, c1, c2)
	mx.Log(mesh)

	m := mx.NewQuant(mx.VECTOR, "m", "", mesh)
	mx.Log(m)
}
