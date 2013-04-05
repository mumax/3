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

	N0, N1, N2 := 1, 64, 128
	c := 1.
	mesh := data.NewMesh(N0, N1, N2, c/2, c*2, c)

	m := cuda.NewSlice(3, mesh)
	conv := cuda.NewDemag(mesh)
	cuda.Memset(m, 1, 1, 1)

	B := cuda.NewSlice(3, mesh)
	Bsat := 1.
	vol := data.NilSlice(1, mesh)
	conv.Exec(B, m, vol, Bsat)
	out := B.HostCopy()

	bx := out.Vectors()[0][N0/2][N1/2][N2/2]
	by := out.Vectors()[1][N0/2][N1/2][N2/2]
	bz := out.Vectors()[2][N0/2][N1/2][N2/2]
	fmt.Println("demag tensor:", bx, by, bz)
	check(bx, -1)
	check(by, 0)
	check(bz, 0)
	fmt.Println("OK")
}

func check(have, want float32) {
	if math.Abs(float64(have-want)) > 1e-2 {
		log.Fatal("error too large: want ", want, " have ", have)
	}
}
