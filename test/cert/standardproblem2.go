// +build ignore

package main

import (
	. "code.google.com/p/mx3/engine"
	"math"
)

func main() {
	Init()
	defer Close()

	Msat = Const(800e3)
	Aex = Const(13e-12)
	Alpha = Const(1.0)
	lex := math.Sqrt(Aex() / (0.5 * Mu0 * Msat() * Msat()))
	I := 10.
	d := I * lex
	Sz, Sy, Sx := 0.1*d, d, 5*d

	Nz, Ny, Nx := 1, 1, 1
	for Sy/float64(Ny) > 0.75*lex {
		Ny *= 2
	}
	for Sx/float64(Nx) > 0.75*lex {
		Nx *= 2
	}

	SetMesh(Nx, Ny, Nz, Sx/float64(Nx), Sy/float64(Ny), Sz/float64(Nz))

	SetMUniform(1, 1, 0)

	M.Autosave(1e-16)
	Table.Autosave(0.1e-9)
	Run(5e-9)
	M.Save()

}

//func expect(have, want float32) {
//	if abs(have-want) > 1e-3 {
//		log.Fatalln("have:", have, "want:", want)
//	}
//}

func abs(x float32) float32 {
	if x < 0 {
		return -x
	}
	return x
}
