//go:build ignore
// +build ignore

package main

import (
	. "github.com/mumax/3/engine"
)

func main() {

	defer InitAndClose()()

	SetGridSize(128, 32, 1)
	SetCellSize(500e-9/128, 125e-9/32, 3e-9)

	Msat.Set(800e3)
	Aex.Set(13e-12)
	Alpha.Set(0.02)
	M.Set(Uniform(1, .1, 0))

	AutoSave(&M, 100e-12)
	TableAdd(MaxTorque)
	TableAutoSave(5e-12)

	Relax()

	// reversal
	B_ext.Set(Vector(-24.6e-3, 4.3e-3, 0))
	Run(1e-9)
	TOL := 1e-3
	ExpectV("m", M.Average(), Vector(-0.9846124053001404, 0.12604089081287384, 0.04327124357223511), TOL)
}
