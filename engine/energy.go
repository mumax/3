package engine

import (
	"code.google.com/p/mx3/cuda"
	"log"
	"path"
	"reflect"
	"runtime"
)

var (
	energyTerms []func() float64 // terms of total energy, all terms should be registered here.
	TotalEnergy = newGetfunc(1, "E", "J", func() []float64 {
		return []float64{GetTotalEnergy()}
	})
)

func init() {
	world.ROnly("energy", &TotalEnergy)
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

type getfunc struct {
	info
	get func() []float64
}

func newGetfunc(nComp int, name, unit string, get func() []float64) getfunc {
	return getfunc{info{nComp, name, unit}, get}
}

func (g *getfunc) GetVec() []float64 {
	return g.get()
}
