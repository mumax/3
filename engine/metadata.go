package engine

/*

The metadata layer wraps micromagnetic basis functions (e.g. func SetDemagField())
in objects that provide:

- additional information (Name, Unit, ...) used for saving output,
- additional methods (Comp, Region, ...) handy for input scripting.

*/

import (
	"fmt"
	"github.com/mumax/3/cuda"
	"github.com/mumax/3/data"
)

// TODO
// Slice() ->  EvalTo(dst)

// The Info interface defines the bare minimum methods a quantity must implement
// to be accessible for scripting and I/O.
type Info interface {
	Name() string // number of components (scalar, vector, ...)
	Unit() string // name used for output file (e.g. "m")
	NComp() int   // unit, e.g. "A/m"
}

// info provides an Info implementation intended for embedding.
type info struct {
	nComp int
	name  string
	unit  string
}

func makeInfo(nComp int, name, unit string) info {
	return info{nComp: nComp, name: name, unit: unit}
}

func (i *info) Name() string { return i.name }
func (i *info) Unit() string { return i.unit }
func (i *info) NComp() int   { return i.nComp }

// an outputValue is an outputable  scalar or vector value
// that has no space-dependence. E.g. averaged magnetization, total energy.
type outputValue interface {
	Info
	average() []float64 // TODO: rename
}

type ScalarValue struct {
	outputValue
}

type VectorValue struct {
	outputValue
}

func NewScalarValue(name, unit, desc string, f func() float64) ScalarValue {
	g := func() []float64 { return []float64{f()} }
	v := ScalarValue{&getFunc{makeInfo(1, name, unit), g}}
	Export(v, desc)
	return v
}

func (s ScalarValue) Get() float64 {
	return s.average()[0]
}

// wraps a func to make it a quantity
// unifies getScalar and getVector
type getFunc struct {
	info
	f func() []float64
}

func (g *getFunc) get() []float64     { return g.f() }
func (g *getFunc) average() []float64 { return g.get() }

func newGetfunc_(nComp int, name, unit, doc_ string, get func() []float64) getFunc {
	return getFunc{makeInfo(nComp, name, unit), get}
}

type GetVector struct{ getFunc }

func (g *GetVector) Get() data.Vector     { return unslice(g.get()) }
func (g *GetVector) Average() data.Vector { return g.Get() }

// INTERNAL
func NewGetVector(name, unit, doc string, get func() []float64) *GetVector {
	g := &GetVector{newGetfunc_(3, name, unit, doc, get)}
	DeclROnly(name, g, cat(doc, unit))
	return g
}

// OutputQuantity represents a space-dependent quantity,
// that can be saved, like M, B_eff or alpha.
type outputField interface {
	Slice() (q *data.Slice, recycle bool) // get quantity data (GPU or CPU), indicate need to recycle
	NComp() int                           // Number of components (1: scalar, 3: vector, ...)
	Name() string                         // Human-readable identifier, e.g. "m", "alpha"
	Unit() string                         // Unit, e.g. "A/m" or "".
	Mesh() *data.Mesh                     // Usually the global mesh, unless this quantity has a special size.
	average() []float64
}

func NewVectorField(name, unit, desc string, f func(dst *data.Slice)) VectorField {
	v := AsVectorField(&callbackOutput{makeInfo(3, name, unit), f})
	Export(v, desc)
	return v
}

func NewScalarField(name, unit string, f func(dst *data.Slice)) ScalarField {
	return AsScalarField(&callbackOutput{makeInfo(1, name, unit), f})
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
type ScalarField struct {
	outputField
}

// AsScalarField promotes a quantity to a ScalarField,
// enabling convenience methods particular to scalars.
func AsScalarField(q outputField) ScalarField {
	if q.NComp() != 1 {
		panic(fmt.Errorf("ScalarField(%v): need 1 component, have: %v", q.Name(), q.NComp()))
	}
	return ScalarField{q}
}

func (s ScalarField) Average() float64         { return s.outputField.average()[0] }
func (s ScalarField) Region(r int) ScalarField { return AsScalarField(inRegion(s.outputField, r)) }

// VectorField is a Quantity guaranteed to have 3 components.
// Provides convenience methods particular to vectors.
type VectorField struct {
	outputField
}

// AsVectorField promotes a quantity to a VectorField,
// enabling convenience methods particular to vectors.
func AsVectorField(q outputField) VectorField {
	if q.NComp() != 3 {
		panic(fmt.Errorf("VectorField(%v): need 3 components, have: %v", q.Name(), q.NComp()))
	}
	return VectorField{q}
}

func (v VectorField) Average() data.Vector     { return unslice(v.outputField.average()) }
func (v VectorField) Region(r int) VectorField { return AsVectorField(inRegion(v.outputField, r)) }
func (v VectorField) Comp(c int) ScalarField   { return AsScalarField(Comp(v.outputField, c)) }
