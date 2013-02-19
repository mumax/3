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

	mhost := m.Data().HostCopy()
	m_ := mhost.Vectors()
	r := float64(N) / 4
	for i := 0; i < N; i++ {
		x := c * (float64(i) + 0.5 - float64(N/2))
		for j := 0; j < N; j++ {
			y := c * (float64(j) + 0.5 - float64(N/2))
			for k := 0; k < N; k++ {
				z := c * (float64(k) + 0.5 - float64(N/2))
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
	B.ReadNext(B.Mesh().NCell()).HostCopy()
	B.ReadDone()
}
