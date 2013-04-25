// +build ignore

package main

// Test Exchange Mask set to uniform,
// using standard problem 4.

import (
	. "code.google.com/p/mx3/engine"
	"fmt"
	"log"
	"math"
)

func main() {
	Init()
	defer Close()

	const (
		Nz, Ny, Nx = 1, 32, 128
		Sz, Sy, Sx = 3e-9, 125e-9, 500e-9
		cz, cy, cx = Sz / Nz, Sy / Ny, Sx / Nx
	)
	SetMesh(Nx, Ny, Nz, cx, cy, cz)

	Alpha = Const(1.0)
	Msat = Const(800e3)
	Aex = Const(13e-12 * 2) // * 2 here...
	M.Set(Uniform(1, .1, 0))
	ExMask.SetAll(X, 0.5)
	ExMask.SetAll(Y, 0.5)
	ExMask.SetAll(Z, 0.5)

	Run(3e-9)

	m := M.Average()
	fmt.Println("relaxed m:", m[X], m[Y], m[Z])
	expect(m[Z], 0)
	expect(m[Y], 0.12528)
	expect(m[X], 0.96696)
	fmt.Println("OK")

}

func expect(have, want float64) {
	if math.Abs(have-want) > 1e-3 {
		log.Fatalln("have:", have, "want:", want)
	}
}
