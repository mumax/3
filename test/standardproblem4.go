package main

import (
	. "github.com/mumax/3/engine"
	. "github.com/mumax/3/mainpkg"
)

func main() {

	Init()
	defer Close()

	SetGridSize(128, 32, 1)
	SetCellSize(500e-9/128, 125e-9/32, 3e-9)

	Msat.Set(800e3)
	Aex.Set(13e-12)
	M.Set(Uniform(1, .1, 0))
	AutoSave(M, 10e-12)

	TableAdd(MaxTorque)
	TableAutosave(5e-12)

	// relax
	Alpha.Set(3)
	MaxErr.Set(1e-4)
	RunWhile(func() bool { return MaxTorque.Get() > 1e-4 })
	Run(1e-9)
	m_ := Average(M)
	Expect("mx", m_[0], 0.96696, 1e-3)
	Expect("my", m_[1], 0.12528, 1e-3)
	Expect("mz", m_[2], 0, 1e-3)

	// reversal
	Alpha.Set(0.02)

	B_ext.Set(Vector(-24.6E-3, 4.3E-3, 0))
	Run(1e-9)
	m_ = Average(M)
	Expect("mx", m_[0], -0.9845, 1e-3)
	Expect("my", m_[1], 0.1268, 1e-3)
	Expect("mz", m_[2], 0.0432, 1e-3)

}
