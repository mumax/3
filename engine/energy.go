package engine

import (
	"github.com/mumax/3/cuda"
	"github.com/mumax/3/data"
)

var (
	energyTerms []func() float64    // all contributions to total energy
	edensTerms  []func(*data.Slice) // all contributions to total energy density (add to dst)
	E_total     = NewGetScalar("E_total", "J", "Total energy", GetTotalEnergy)
	Edens_total setter
)

func init() {
	Edens_total.init(SCALAR, "Edens_total", "J/m3", "Total energy density", SetTotalEdens)
}

// add energy term to global energy
func registerEnergy(term func() float64, dens func(*data.Slice)) {
	energyTerms = append(energyTerms, term)
	edensTerms = append(edensTerms, dens)
}

// Returns the total energy in J.
func GetTotalEnergy() float64 {
	E := 0.
	for _, f := range energyTerms {
		E += f()
	}
	return E
}

func SetTotalEdens(dst *data.Slice) {
	cuda.Zero(dst)
	for _, addTerm := range edensTerms {
		addTerm(dst)
	}
}

// vector dot product
func dot(a, b Slicer) float64 {
	A, recyA := a.Slice()
	if recyA {
		defer cuda.Recycle(A)
	}
	B, recyB := b.Slice()
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
