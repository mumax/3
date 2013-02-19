// +build ignore

package main

import (
	"code.google.com/p/mx3/cuda"
	"code.google.com/p/mx3/data"
)

func main() {
	cuda.Init()

	N := 16
	c := 1e-3
	mesh := data.NewMesh(N, N, N, c, c, c)

	m := cuda.NewQuant(3, mesh)
	conv := cuda.NewDemag(m)
	m_ := m.Data().Vectors()
	for i := 0; i < N; i++ {
		for j := 0; j < N; j++ {
			for k := 0; k < N; k++ {
				x := float64(i)
			}
		}
	}

	conv.Exec()
	B := conv.Output().Data().HostCopy()
}
