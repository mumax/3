package engine

import (
	"fmt"
	"github.com/mumax/3/cuda"
	"github.com/mumax/3/data"
)

// TODO
// Slice() ->  EvalTo(dst)

// Quantity represents a space-dependent quantity,
// Like M, B_eff or alpha.
type OutputQuantity interface {
	Slice() (q *data.Slice, recycle bool) // get quantity data (GPU or CPU), indicate need to recycle
	NComp() int                           // Number of components (1: scalar, 3: vector, ...)
	Name() string                         // Human-readable identifier, e.g. "m", "alpha"
	Unit() string                         // Unit, e.g. "A/m" or "".
	Mesh() *data.Mesh                     // Usually the global mesh, unless this quantity has a special size.
	average() []float64
}

func NewQuantity(nComp int, name, unit string, f func(dst *data.Slice)) OutputQuantity {
	return &callbackOutput{info{nComp, name, unit}, f}
}

func NewVectorOutput(name, unit string, f func(dst *data.Slice)) VectorOutput {
	return AsVectorOutput(NewQuantity(3, name, unit, f))
}

func NewScalarOutput(name, unit string, f func(dst *data.Slice)) ScalarOutput {
	return AsScalarOutput(NewQuantity(1, name, unit, f))
}

type callbackOutput struct {
	info
	call func(*data.Slice)
}

func (c *callbackOutput) Mesh() *data.Mesh   { return Mesh() }
func (c *callbackOutput) average() []float64 { return qAverageUniverse(c) }

// Calculates and returns the quantity.
// recycle is true: slice needs to be recycled.
func (q *callbackOutput) Slice() (s *data.Slice, recycle bool) {
	buf := cuda.Buffer(q.NComp(), q.Mesh().Size())
	cuda.Zero(buf)
	q.call(buf)
	return buf, true
}

// ScalarField is a Quantity guaranteed to have 1 component.
// Provides convenience methods particular to scalars.
type ScalarOutput struct {
	OutputQuantity
}

// AsScalarField promotes a quantity to a ScalarField,
// enabling convenience methods particular to scalars.
func AsScalarOutput(q OutputQuantity) ScalarOutput {
	if q.NComp() != 1 {
		panic(fmt.Errorf("ScalarField(%v): need 1 component, have: %v", q.Name(), q.NComp()))
	}
	return ScalarOutput{q}
}

func (s ScalarOutput) Average() float64          { return s.OutputQuantity.average()[0] }
func (s ScalarOutput) Region(r int) ScalarOutput { return AsScalarOutput(inRegion(s.OutputQuantity, r)) }

// VectorField is a Quantity guaranteed to have 3 components.
// Provides convenience methods particular to vectors.
type VectorOutput struct {
	OutputQuantity
}

// AsVectorField promotes a quantity to a VectorField,
// enabling convenience methods particular to vectors.
func AsVectorOutput(q OutputQuantity) VectorOutput {
	if q.NComp() != 3 {
		panic(fmt.Errorf("VectorField(%v): need 3 components, have: %v", q.Name(), q.NComp()))
	}
	return VectorOutput{q}
}

func (v VectorOutput) Average() data.Vector      { return unslice(v.OutputQuantity.average()) }
func (v VectorOutput) Region(r int) VectorOutput { return AsVectorOutput(inRegion(v.OutputQuantity, r)) }
func (v VectorOutput) Comp(c int) ScalarOutput   { return AsScalarOutput(Comp(v.OutputQuantity, c)) }
