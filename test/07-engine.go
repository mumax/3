// +build ignore

package main

import . "code.google.com/p/mx3/engine"

func main() {
	Init()
	defer Close()

	SetMesh(1024, 1024, 1, 4e-9, 4e-9, 4e-9)

	SetM(1, 1, 0)
	Msat = Const(800e3)
	Aex = Const(13e-12)
	Alpha = Const(1)

	Exch.Autosave(0.0000001e-9)
	Steps(10)
}
