// +build ignore

package main

// Micromagnetic standard problem 4b according to
// http://www.ctcms.nist.gov/~rdm/mumag.org.html

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
	Aex = Const(13e-12)
	M.Set(Uniform(1, .1, 0))

	Table.Autosave(1e-12)
	Run(3e-9)

	m := M.Average()
	fmt.Println("relaxed m:", m[X], m[Y], m[Z])
	expect(m[Z], 0)
	expect(m[Y], 0.12528)
	expect(m[X], 0.96696)
	fmt.Println("OK")

	Alpha = Const(0.02)
	B_ext = ConstVector(-35.5E-3, -6.3E-3, 0)

	Time = 0
	M.Autosave(50e-12)
	Run(1e-9)

	m = M.Average()
	fmt.Println("final m:", m[X], m[Y], m[Z])
	expect(m[Z], -0.00569)
	expect(m[Y], -0.1186)
	expect(m[X], -0.9687)
	fmt.Println("OK")
}

func expect(have, want float64) {
	if math.Abs(have-want) > 1e-3 {
		log.Fatalln("have:", have, "want:", want)
	}
}
