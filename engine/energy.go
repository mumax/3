package engine

import (
	"code.google.com/p/mx3/cuda"
	"log"
	"path"
	"reflect"
	"runtime"
)

var (
	energyTerms []func() float64 // registers total energy terms
	//E_total     = NewGetScalar("Energy", "J", GetTotalEnergy)
)

func init() {
	//DeclROnly("E_total", &E_total, "Total energy (J)")
}

// add energy term to global energy
func registerEnergy(term func() float64) {
	name := path.Ext(runtime.FuncForPC(reflect.ValueOf(term).Pointer()).Name())
	name = name[1:len(name)]
	log.Println("total energy includes", name)
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
		defer cuda.RecycleBuffer(A)
	}
	B, recyB := b.Get()
	if recyB {
		defer cuda.RecycleBuffer(B)
	}
	return float64(cuda.Dot(A, B))
}

// volume of one cell in m3
func cellVolume() float64 {
	c := Mesh().CellSize()
	return c[0] * c[1] * c[2]
}
