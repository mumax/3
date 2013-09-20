package engine

import "github.com/mumax/3/cuda"

var (
	energyTerms []func() float64 // registers total energy terms
	E_total     = NewGetScalar("E_total", "J", "Total energy", GetTotalEnergy)
)

// add energy term to global energy
func registerEnergy(term func() float64) {
	energyTerms = append(energyTerms, term)
}

// Returns the total energy in J.
func GetTotalEnergy() float64 {
	E := 0.
	for _, f := range energyTerms {
		E += f()
	}
	return E
}

// vector dot product
func dot(a, b Getter) float64 {
	A, recyA := a.Get()
	if recyA {
		defer cuda.Recycle(A)
	}
	B, recyB := b.Get()
	if recyB {
		defer cuda.Recycle(B)
	}
	return float64(cuda.Dot(A, B))
}

// volume of one cell in m3
func cellVolume() float64 {
	c := Mesh().CellSize()
	return c[0] * c[1] * c[2]
}
