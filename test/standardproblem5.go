// +build ignore

package main

// Micromagnetic standard problem 5
// as proposed by M. Najafi et al., JAP 105, 113914 (2009).
// Reference solution by mumax2.

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
		Nz, Ny, Nx = 4, 32, 32
		Sz, Sy, Sx = 10e-9, 100e-9, 100e-9
		cz, cy, cx = Sz / Nz, Sy / Ny, Sx / Nx
	)
	SetMesh(Nx, Ny, Nz, cx, cy, cz)

	Alpha = Const(3)
	Msat = Const(800e3)
	Aex = Const(13e-12)
	M.Upload(Vortex(1, 1))

	Run(1e-9)

	//Table.Autosave(10e-12)
	//M.Autosave(50e-12)
	//STT.Autosave(50e-12)

	Alpha = Const(0.1)
	Xi = Const(0.05)
	SpinPol = Const(1)
	J = ConstVector(1e12, 0, 0)

	Run(1e-9)

	m := M.Average()
	fmt.Println("m (1ns):", m[0], m[1], m[2])
	expect(m[0], -0.239191)
	expect(m[1], -0.099219)
	expect(m[2], 0.0228132)
	fmt.Println("OK")
}

func expect(have, want float64) {
	if math.Abs(have-want) > 1e-4 {
		log.Fatalln("have:", have, "want:", want)
	}
}
