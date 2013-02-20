// +build ignore

package main

import (
	"code.google.com/p/mx3/cuda"
	"code.google.com/p/mx3/data"
	"fmt"
)

func main() {
	cuda.Init()

	N0, N1, N2 := 1, 32, 64
	c := 1.
	mesh := data.NewMesh(N0, N1, N2, c, c, c)

	m := cuda.NewQuant(3, mesh)
	conv := cuda.NewDemag(m)

	mgpu := m.WriteNext(m.Mesh().NCell())
	cuda.Memset(mgpu, 1, 0, 0)
	m.WriteDone()

	B := conv.Output().NewReader()
	conv.Exec()
	out := B.ReadNext(B.Mesh().NCell()).HostCopy()
	B.ReadDone()
	data.MustWriteFile("B.dump", out, 0)
	bx := out.Vectors()[0][N0/2][N1/2][N2/2]
	by := out.Vectors()[1][N0/2][N1/2][N2/2]
	bz := out.Vectors()[2][N0/2][N1/2][N2/2]
	fmt.Println(bx, by, bz)
}
