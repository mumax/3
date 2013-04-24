// +build ignore

package main

// Test energy calculation.

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
		Nz, Ny, Nx = 1, 32, 32
		Sz, Sy, Sx = 10e-9, 100e-9, 100e-9
		cz, cy, cx = Sz / Nz, Sy / Ny, Sx / Nx
	)
	SetMesh(Nx, Ny, Nz, cx, cy, cz)

	Alpha = Const(3)
	Msat = Const(800e3)
	Aex = Const(13e-12)
	M.Upload(Vortex(1, 1))
	B_ext = ConstVector(1e-3, 0, 0)

	Run(1e-9)

	B_ext = ConstVector(0, 0, 0)
	Solver.MaxErr = 1e-6
	Alpha = Const(1e-4)

	//	26015.5 // 1mT
	//	26019.7 // 0mT

	for i := 0; i < 100; i++ {
		Run(0.01e-9)
		fmt.Println(ExchangeEnergy(), DemagEnergy(), ExchangeEnergy()+DemagEnergy())
	}

	Alpha = Const(3)
	for i := 0; i < 100; i++ {
		Run(0.01e-9)
		fmt.Println(ExchangeEnergy(), DemagEnergy(), ExchangeEnergy()+DemagEnergy())
	}
}

func expect(have, want float64) {
	if math.Abs(have-want) > 1e-4 {
		log.Fatalln("have:", have, "want:", want)
	}
}
