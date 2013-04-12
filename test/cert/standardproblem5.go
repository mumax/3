// +build ignore

package main

// Micromagnetic standard proplem no. 5
// As proposed by M. Najafi et al., JAP 105, 113914 (2009).

import (
	. "code.google.com/p/mx3/engine"
	"log"
	"math"
)

func main() {
	Init()
	defer Close()

	const (
		Nz, Ny, Nx = 1, 32, 32
		Sz, Sy, Sx = 10e-9, 100e-9, 100e-9
		cz, cy, cx = Sz / Nz, Sy / Ny, Sx / Nx
	)
	SetMesh(Nx, Ny, Nz, cx, cy, cz)

	Alpha = Const(1.0)
	Msat = Const(800e3)
	Aex = Const(13e-12)
	M.Upload(Vortex(1, 1))

	Table.Autosave(1e-12)
	Run(2e-9)

	M.Autosave(50e-12)
	STT.Autosave(50e-12)

	Solver.Dt_si = 0.1e-12
	Solver.Fixdt = true

	Alpha = Const(0.1)
	Xi = Const(0.05)
	SpinPol = Const(1)
	J = ConstVector(1e12, 0, 0)

	Run(10e-9)

}

func expect(have, want float64) {
	if math.Abs(have-want) > 1e-3 {
		log.Fatalln("have:", have, "want:", want)
	}
}

//Nx = 32
//Ny = 32
//Nz = 1
//
//sizeX = 100e-9
//sizeY = 100e-9
//sizeZ = 10e-9
//
//setv('Msat', 800e3)
//setv('Aex', 1.3e-11)
//setv('alpha', 1.0)
//
//
//
//setv('alpha', 0.1)
//
//setv('xi',0.05)
//setv('polarisation',1.0)
//
//setv('j', [1e12, 0, 0])
//
//run(15.0e-9)
