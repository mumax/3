// package mag provides magnetism-specific constants and the demag kernel.
package mag

import "math"

const (
	Mu0 = 4 * math.Pi * 1e-7 // Permeability of vacuum in Tm/A
	MuB = 9.2740091523e-24   // Bohr magneton in J/T
	Kb  = 1.380650424e-23    // Boltzmann's constant in J/K
	Qe  = 1.60217646e-19     // Electron charge in C
)
