// +build ignore

package main

/*
 In this example we drive a vortex wall in Permalloy by spin-transfer torque.
 We set up a post-step function that makes the simulation box "follow" the domain
 wall. By removing surface charges at the left and right ends, we mimic an infintely
 long wire.
*/

import (
	. "code.google.com/p/mx3/engine"
	"math/rand"
)

func main() {
	Init()
	defer Close()

	// Geometry
	Nx, Ny, Nz := 256, 64, 1
	cellx, celly, cellz := 3e-9, 3e-9, 30e-9
	SetMesh(Nx, Ny, Nz, cellx, celly, cellz)

	// Material parameters
	Msat = Const(860e3)
	Aex = Const(13e-12)
	Xi = Const(0.1)
	SpinPol = Const(0.56)

	// Initial magnetization close to vortex wall
	M.SetRegion(0, 0, 0, Nx/2, Ny, Nz, Uniform(1, 0, 0))          // left half:  ->
	M.SetRegion(Nx/2, 0, 0, Nx, Ny, Nz, Uniform(-1, 0, 0))        // right half: <-
	M.SetRegion(Nx/2-Ny/2, 0, 0, Nx/2+Ny/2, Ny, Nz, Vortex(1, 1)) // center: vortex

	// Remove surface charges from left (mx=1) and right (mx=-1) sides
	// to mimic infinitely long wire.
	RemoveLRSurfaceCharge(1, -1)
	// Set post-step function that centers simulation window on domain wall.
	PostStep(centerInplaneWall)

	Alpha = Const(3)    // high damping for fast relax
	Run(5e-9)           // relax
	Alpha = Const(0.02) // restore normal damping

	// Schedule output
	M.Autosave(100e-12)
	Table.Autosave(10e-12)

	// Run the simulation with current through the sample
	J = ConstVector(-8e12, 0, 0)
	Run(10e-9)
}

// Shift the magnetization to the left in order to keep mx close zero.
// So the simulation window remains centered on the right-moving domain wall.
func centerInplaneWall() {
	mx := M.Average()[X]
	if mx > 0.01 {
		M.Shift(-1, 0, 0) // 1 cell to the left
		makeDefects(254)  // new defects near right border
	}
}

// randomly remove a cell at the right edge of the wire to induce "holes".
func makeDefects(x int) {
	if rand.Intn(5) == 0 {
		mx, my, mz := 0., 0., 0.
		ix, iy, iz := x, rand.Intn(64), 0
		M.SetCell(ix, iy, iz, mx, my, mz)
	}
}
