package engine

import (
	"github.com/mumax/3/data"
)

// wraps a func to make it a quantity
// unifies getScalar and getVector
type getFunc struct {
	info
	get func() []float64
}

func (g *getFunc) average() []float64 { return g.get() }

func newGetfunc_(nComp int, name, unit, doc_ string, get func() []float64) getFunc {
	return getFunc{Info(nComp, name, unit), get}
}

type GetScalar struct{ getFunc }
type GetVector struct{ getFunc }

func (g *GetScalar) Get() float64     { return g.get()[0] }
func (g *GetScalar) Average() float64 { return g.Get() }

func (g *GetVector) Get() data.Vector     { return unslice(g.get()) }
func (g *GetVector) Average() data.Vector { return g.Get() }

// INTERNAL
func NewGetScalar(name, unit, doc string, get func() float64) *GetScalar {
	g := &GetScalar{newGetfunc_(1, name, unit, doc, func() []float64 {
		return []float64{get()}
	})}
	DeclROnly(name, g, cat(doc, unit))
	return g
}

// INTERNAL
func NewGetVector(name, unit, doc string, get func() []float64) *GetVector {
	g := &GetVector{newGetfunc_(3, name, unit, doc, get)}
	DeclROnly(name, g, cat(doc, unit))
	return g
}
