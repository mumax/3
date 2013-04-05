// +build ignore

package main

import (
	"code.google.com/p/mx3/cuda"
	"code.google.com/p/mx3/data"
	"fmt"
	"log"
	"math"
)

func main() {
	cuda.Init()

	N0, N1, N2 := 16, 16, 16
	c := 1.
	mesh := data.NewMesh(N0, N1, N2, c, c, c)

	m := cuda.NewSlice(3, mesh)
	conv := cuda.NewDemag(mesh)

	mhost := m.HostCopy()
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
					m_[1][i][j][k] = 2
					m_[2][i][j][k] = 3
				}
			}
		}
	}

	data.Copy(m, mhost)

	B := cuda.NewSlice(3, mesh)
	conv.Exec(B, m, data.NilSlice(1, mesh), 1)
	out := B.HostCopy()

	bx := out.Vectors()[0][N0/2][N1/2][N2/2]
	by := out.Vectors()[1][N0/2][N1/2][N2/2]
	bz := out.Vectors()[2][N0/2][N1/2][N2/2]
	fmt.Println("demag tensor:", bx, by/2, bz/3)
	check(bx, -1./3.)
	check(by, -2./3.)
	check(bz, -3./3.)
	fmt.Println("OK")
}

func check(have, want float32) {
	if math.Abs(float64(have-want)) > 1e-3 {
		log.Fatal("error too large: want ", want, " have ", have)
	}
}
