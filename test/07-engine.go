// +build ignore

package main

import . "code.google.com/p/mx3/engine"

func main() {
	Init()

	SetMesh(128, 32, 1, 500e-9/128, 125e-9/32, 3e-9)

	SetM(1, 1, 0)
	Msat = Const(800e3)
	Aex = Const(13e-12)
	Alpha = Const(1)

	Run(2e-9)
}
