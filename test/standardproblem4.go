package main

import (
	. "github.com/mumax/3/engine"
	. "github.com/mumax/3/init"
)

func main() {

	Init()
	SetOD("standardproblem4.out", true)
	defer Close()

	SetGridSize(128, 32, 1)
	SetCellSize(500e-9/128, 125e-9/32, 3e-9)

	Msat.Set(800e3)
	Aex.Set(13e-12)
	M.Set(Uniform(1, .1, 0))
	AutoSave(&M, 100e-12)

	TableAdd(MaxTorque)
	TableAutoSave(5e-12)

	// relax
	Alpha.Set(3)
	Solver.MaxErr = 1e-4
	RunWhile(func() bool { return MaxTorque.Get() > 1e-4 })
	Run(1e-9)

	// reversal
	Alpha.Set(0.02)

	B_ext.Set(Vector(-24.6E-3, 4.3E-3, 0))
	Run(1e-9)
	ExpectV("m", M.Average(), Vector(-0.9845, 0.1268, 0.0432), 1e-3)
}
