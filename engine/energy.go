package engine

import (
	"code.google.com/p/mx3/cuda"
)

// Returns the current exchange energy in Joules.
// Note: the energy is defined up to an arbitrary constant,
// ground state energy is not necessarily zero or comparable
// to other simulation programs.
func ExchangeEnergy() float64 {
	return -0.5 * Volume() * dot(&M_full, &B_exch) / Mu0
}

// Returns the current demag energy in Joules.
func DemagEnergy() float64 {
	return -0.5 * Volume() * dot(&M_full, &B_demag) / Mu0
}

func dot(a, b GPU_Getter) float64 {
	A, recyA := a.GetGPU()
	if recyA {
		defer cuda.RecycleBuffer(A)
	}
	B, recyB := b.GetGPU()
	if recyB {
		defer cuda.RecycleBuffer(B)
	}
	return float64(cuda.Dot(A, B))
}

func Volume() float64 {
	c := Mesh().CellSize()
	return c[0] * c[1] * c[2]
}
