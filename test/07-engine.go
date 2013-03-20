// +build ignore

package main

import . "code.google.com/p/mx3/engine"

func main() {
	Init()
	defer Close()

	SetMesh(256, 64, 1, 500e-9/256, 125e-9/64, 3e-9)

	SetM(1, 1, 0)
	Msat = Const(800e3)
	Aex = Const(13e-12)
	Alpha = Const(1)

	Exch.Autosave(0.1e-9)
	Table.Autosave(0.01e-9)
	Run(1e-9)
}
