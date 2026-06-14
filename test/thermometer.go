//go:build ignore
// +build ignore

/*
Checks if the measured temperature in a ferromagnetic PMA film is equal to the input temperature.
We measure the temperature with the thermometer derived in PHYSICAL REVIEW E 82, 031111 (2010):

    T = (Vcell*Msat)/(2*kB) * <Σ||m x h||^2> / <Σ m.h >     [1]

The expectation values <...> are calculated by taking time averages.
The sums Σ... are taken over the different cells.

The input temperature is chosen to be 177K.
We allow an error smaller than 5K.

NOTE:

The exchange energy in MuMax3 is shifted by a constant with respect to atomistic simulations.
Due to this difference, we need to add the following constant value to the divisor of [1]:

    shift = 2 * (Aex/Msat) * NCell * ( 2/Δx² + 2/Δy² )

*/

package main

import (
	"github.com/mumax/3/cuda"
	. "github.com/mumax/3/engine"
)

const kB = 1.38064852e-23 // Boltzmann constant

func main() {

	defer InitAndClose()()

	// Prepare the PMA film
	Eval(`
        	SetGridSize(128, 128, 1)
        	SetCellSize(4e-9, 4e-9, 4e-9)
		SetPBC(1,1,0)
        	Msat = 580e3
        	Aex = 15e-12
        	AnisU = Vector(0, 0, 1)
        	Ku1 = 0.6e6
        	Alpha = 0.1
        	Temp = 177
        	M = Uniform(0, 0, -1)
        	Run(1e-10)
	`)

	m := M.Buffer()
	h := cuda.Buffer(3, m.Size())
	mxh := cuda.Buffer(3, m.Size())

	cs := Mesh().CellSize()
	Vcell := cs[X] * cs[Y] * cs[Z]
	shift := 2 * Aex.GetRegion(0) / Msat.Average() * float64(Mesh().NCell()) * (2/(cs[X]*cs[X]) + 2/(cs[Y]*cs[Y]))

	// update the time averages in numerator and divisor of [1] in each step from now on
	divisor := 0.0
	numerator := 0.0
	nstep := 0.0
	PostStep(func() {
		nstep += 1
		SetDemagField(h)
		AddExchangeField(h)
		AddAnisotropyField(h)
		cuda.CrossProduct(mxh, m, h)
		divisor = ((nstep-1)*divisor + float64(cuda.Dot(m, h))) / nstep
		numerator = ((nstep-1)*numerator + float64(cuda.Dot(mxh, mxh))) / nstep
	})

	Run(1e-10)

	temperature := (Vcell * Msat.Average() / (2 * kB)) * numerator / (divisor + shift) // [1]
	Expect("temperature", temperature, Temp.GetRegion(0), 5)
}
