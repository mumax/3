package main

import (
	"github.com/mumax/3/cuda"
	. "github.com/mumax/3/engine"
)

func main() {

	cuda.Init(0)
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
	RunWhile(func() bool { return MaxTorque.Get() > 1e-4 })
	Run(1e-9)

	// reversal
	Alpha.Set(0.02)

	B_ext.Set(Vector(-24.6E-3, 4.3E-3, 0))
	Run(1e-9)
	ExpectV("m", M.Average(), Vector(-0.9846177101135254, 0.12599533796310425, 0.043271854519844055), 1e-3)
}
