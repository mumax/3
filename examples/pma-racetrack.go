package main

/*
 In this example we drive a domain wall in PMA material by spin-transfer torque.
 We set up a post-step function that makes the simulation box "follow" the domain
 wall. Like this, only a small number of cells is needed to simulate an infinitely
 long magnetic wire.
*/

import . "code.google.com/p/mx3/engine"

func main() {
	Init()
	defer Close()

	// Geometry
	Nx, Ny, Nz := 128, 256, 1
	cellx, celly, cellz := 2e-9, 2e-9, 1e-9
	SetMesh(Nx, Ny, Nz, cellx, celly, cellz)

	// Material parameters
	Msat = Const(600e3)
	Aex = Const(10e-12)
	Alpha = Const(1)
	Ku1 = ConstVector(0, 0, 0.59E6)

	// Initial magnetization
	M.Set(TwoDomain(0, 0, 1, 1, 1, 0, 0, 0, -1)) // up-down domains with wall between Bloch and Néél type
	Run(0.1e-9)                                  // relax

	// Set post-step function that centers simulation window on domain wall.
	PostStep(centerPMAWall)

	// New parameters
	Alpha = Const(0.02)
	Xi = Const(0.2)
	SpinPol = Const(0.5)
	J = ConstVector(1e13, 0, 0)

	// Schedule output
	M.Autosave(100e-12)
	Table.Autosave(10e-12)

	// Run for 1ns
	Run(1e-9)
}

// Shift the magnetization to the left or right in order to keep mz close zero.
// Thus moving an up-down domain wall to the center of the simulation box.
func centerPMAWall() {
	mz := M.Average()[Z]
	if mz > 0.01 {
		M.Shift(-1, 0, 0) // 1 cell to the left
	}
	if mz < -0.01 {
		M.Shift(1, 0, 0) // 1 cell to the right
	}
}
