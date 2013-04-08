// +build ignore

package main

import (
	. "code.google.com/p/mx3/engine"
	"code.google.com/p/mx3/web"
)

func main() {
	Init()
	defer Close()

	const Nx, Ny = 128, 32
	SetMesh(Nx, Ny, 1, 500e-9/Nx, 125e-9/Ny, 3e-9)

	Msat = Const(800e3)
	Aex = Const(13e-12)
	Alpha = Const(1)
	SetMUniform(1, 1, 0)

	web.GoServe(":8080")

	Run(1e-9)

	f := 1.0
	Alpha = Const(0.02)

	for {
		B_ext = ConstVector(f*24.6E-3, -f*4.3E-3, 0)
		Run(2e-9)
		f = -f
	}

}
