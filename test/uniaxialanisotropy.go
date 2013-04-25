// +build ignore

package main

// This test lets uniaxial anisotropy compete against the external field.
// Reference values obtained with OOMMF.

import (
	. "code.google.com/p/mx3/engine"
	"fmt"
	"log"
	"math"
)

func main() {
	Init()
	defer Close()

	const Nz, Ny, Nx = 1, 64, 64
	const cz, cy, cx = 2e-9, 4e-9, 4e-9

	Msat = Const(1100e3)
	Aex = Const(13e-12)
	Alpha = Const(0.2)
	Ku1 = ConstVector(0.5e6, 0, 0)

	SetMesh(Nx, Ny, Nz, cx, cy, cz)

	M.Set(Uniform(1, 1, 0))
	Table.Autosave(10e-12)

	// Apply some fields and verify the relaxed my agains OOMMF values.
	reference := []float64{0, 0.011, 0.033, 0.110, 0.331}
	for i, by := range []float64{0, 10, 30, 100, 300} {
		By := by * 1e-3
		B_ext = ConstVector(0, By, 0)
		Run(1e-9)
		m := M.Average()
		fmt.Println("By:", By, "T", "my:", m[Y])
		expect(m[Y], reference[i])
	}
	fmt.Println("OK")
}

func expect(have, want float64) {
	if math.Abs(have-want) > 1e-3 {
		log.Fatalln("have:", have, "want:", want)
	}
}
