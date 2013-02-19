// +build ignore

package main

import (
	"code.google.com/p/mx3/cuda"
	"code.google.com/p/mx3/data"
	"fmt"
)

func main() {
	cuda.Init()

	N0, N1, N2 := 32, 32, 32
	c := 1e-3
	mesh := data.NewMesh(N0, N1, N2, c, c, c)

	m := cuda.NewQuant(3, mesh)
	conv := cuda.NewDemag(m)

	mhost := m.Data().HostCopy()
	m_ := mhost.Vectors()
	r := float64(N2) / 2
	for i := 0; i < N0; i++ {
		x := c * (float64(i) + 0.5 - float64(N0)/2)
		for j := 0; j < N1; j++ {
			y := c * (float64(j) + 0.5 - float64(N1)/2)
			for k := 0; k < N2; k++ {
				z := c * (float64(k) + 0.5 - float64(N2)/2)
				if x*x+y*y+z*z < r*r {
					m_[0][i][j][k] = 1
				}
			}
		}
	}

	mgpu := m.WriteNext(m.Mesh().NCell())
	data.Copy(mgpu, mhost)
	m.WriteDone()

	B := conv.Output().NewReader()
	conv.Exec()
	out := B.ReadNext(B.Mesh().NCell()).HostCopy()
	B.ReadDone()
	data.MustWriteFile("B.dump", out, 0)
	b := out.Vectors()[0][N0/2][N1/2][N2/2]
	fmt.Println(b)
}
