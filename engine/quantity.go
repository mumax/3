package engine

import (
	"fmt"
	"github.com/mumax/3/cuda"
	"github.com/mumax/3/data"
)

// Quantity represents a space-dependent quantity,
// Like M, B_eff or alpha.
type Quantity interface {
	Slice() (q *data.Slice, recycle bool) // get quantity data (GPU or CPU), indicate need to recycle
	NComp() int                           // Number of components (1: scalar, 3: vector, ...)
	Name() string                         // Human-readable identifier, e.g. "m", "alpha"
	Unit() string                         // Unit, e.g. "A/m" or "".
	Mesh() *data.Mesh                     // Usually the global mesh, unless this quantity has a special size.
	average() []float64
}

func AsQuantity(nComp int, name, unit, doc string, f func(dst *data.Slice)) Quantity {
	return &callbackQuant{info{nComp, name, unit}, f}
}

type callbackQuant struct {
	info
	call func(*data.Slice)
}

func (c *callbackQuant) Mesh() *data.Mesh   { return Mesh() }
func (c *callbackQuant) average() []float64 { return qAverageUniverse(c) }

// Calculates and returns the quantity.
// recycle is true: slice needs to be recycled.
func (q *callbackQuant) Slice() (s *data.Slice, recycle bool) {
	buf := cuda.Buffer(q.NComp(), q.Mesh().Size())
	q.call(buf)
	return buf, true
}

// ScalarField is a Quantity guaranteed to have 1 component.
// Provides convenience methods particular to scalars.
type ScalarField struct {
	Quantity
}

// AsScalarField promotes a quantity to a ScalarField,
// enabling convenience methods particular to scalars.
func AsScalarField(q Quantity) ScalarField {
	if q.NComp() != 1 {
		panic(fmt.Errorf("ScalarField(%v): need 1 component, have: %v", q.Name(), q.NComp()))
	}
	return ScalarField{q}
}

func (s ScalarField) Average() float64         { return s.Quantity.average()[0] }
func (s ScalarField) Region(r int) ScalarField { return AsScalarField(inRegion(s.Quantity, r)) }

// VectorField is a Quantity guaranteed to have 3 components.
// Provides convenience methods particular to vectors.
type VectorField struct {
	Quantity
}

// AsVectorField promotes a quantity to a VectorField,
// enabling convenience methods particular to vectors.
func AsVectorField(q Quantity) VectorField {
	if q.NComp() != 3 {
		panic(fmt.Errorf("VectorField(%v): need 3 components, have: %v", q.Name(), q.NComp()))
	}
	return VectorField{q}
}

func NewVectorField() {

}

func (v VectorField) Average() data.Vector     { return unslice(v.Quantity.average()) }
func (v VectorField) Region(r int) VectorField { return AsVectorField(inRegion(v.Quantity, r)) }
func (v VectorField) Comp(c int) ScalarField   { return AsScalarField(Comp(v.Quantity, c)) }
