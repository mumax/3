package main

// main program starts a web gui.

import (
	. "code.google.com/p/mx3/engine"
)

func main() {
	Init()
	defer Close()

	SetMesh(32, 32, 1, 1e-9, 1e-9, 1e-9)

	Msat = Const(1000e3)
	Aex = Const(10e-12)
	Alpha = Const(1)
	SetMUniform(1, 1, 0)

	Interactive()

}
