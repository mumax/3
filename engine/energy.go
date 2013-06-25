package engine

import (
	"code.google.com/p/mx3/cuda"
	"log"
	"path"
	"reflect"
	"runtime"
)

// terms of total energy, all terms should be registered here.
var energyTerms []func() float64

// add energy term to global energy
func registerEnergy(term func() float64) {
	name := path.Ext(runtime.FuncForPC(reflect.ValueOf(term).Pointer()).Name())
	name = name[1:len(name)]
	log.Println("total energy includes", name)
	energyTerms = append(energyTerms, term)
}

// Returns the total energy in J.
func TotalEnergy() float64 {
	E := 0.
	for _, f := range energyTerms {
		E += f()
	}
	return E
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
