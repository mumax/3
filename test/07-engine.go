// +build ignore

package main

import . "code.google.com/p/mx3/engine"

func main() {
	Init()
	defer Close()

	const Nx, Ny = 128, 32
	SetMesh(Nx, Ny, 1, 500e-9/Nx, 125e-9/Ny, 3e-9)

	Msat = Const(800e3)
	Aex = Const(13e-12)
	Alpha = Const(1)
	SetM(1, 1, 0)

	B_exch.Autosave(1e-9)
	B_demag.Autosave(1e-9)
	M.Autosave(1e-9)
	B_eff.Autosave(1e-9)
	Torque.Autosave(1e-9)

	Run(5e-9)

	B_ext = ConstVector(-24.6E-3, 4.3E-3, 0)
	Alpha = Const(0.02)
	M.Autosave(0.01e-9)
	Table.Autosave(1e-12)
	Run(1e-9)
}
