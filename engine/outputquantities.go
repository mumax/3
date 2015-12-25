// TODO
// Slice() ->  EvalTo(dst)

package engine

/*
The metadata layer wraps basic micromagnetic functions (e.g. func SetDemagField())
in objects that provide:

- additional information (Name, Unit, ...) used for saving output,
- additional methods (Comp, Region, ...) handy for input scripting.
*/

import (
	"fmt"
	"github.com/mumax/3/cuda"
	"github.com/mumax/3/data"
)

// The Info interface defines the bare minimum methods a quantity must implement
// to be accessible for scripting and I/O.
type Info interface {
	Name() string // number of components (scalar, vector, ...)
	Unit() string // name used for output file (e.g. "m")
	NComp() int   // unit, e.g. "A/m"
}

// info provides an Info implementation intended for embedding in other types.
type info struct {
	nComp int
	name  string
	unit  string
}

func (i *info) Name() string { return i.name }
func (i *info) Unit() string { return i.unit }
func (i *info) NComp() int   { return i.nComp }

// outputValue must be implemented by any scalar or vector quantity
// that has no space-dependence, to make it outputable.
// The space-dependent counterpart is outputField.
// Used internally by ScalarValue and VectorValue.
type outputValue interface {
	Info
	average() []float64 // TODO: rename
}

// outputFunc is an outputValue implementation where a function provides the output value.
// It can be scalar or vector.
// Used internally by NewScalarValue and NewVectorValue.
type valueFunc struct {
	info
	f func() []float64
}

func (g *valueFunc) get() []float64     { return g.f() }
func (g *valueFunc) average() []float64 { return g.get() }

// ScalarValue enhances an outputValue with methods specific to
// a space-independent scalar quantity (e.g. total energy).
type ScalarValue struct {
	outputValue
}

// NewScalarValue constructs an outputable space-independent scalar quantity whose
// value is provided by function f.
func NewScalarValue(name, unit, desc string, f func() float64) *ScalarValue {
	g := func() []float64 { return []float64{f()} }
	v := &ScalarValue{&valueFunc{info{1, name, unit}, g}}
	Export(v, desc)
	return v
}

func (s ScalarValue) Get() float64     { return s.average()[0] }
func (s ScalarValue) Average() float64 { return s.Get() }

// VectorValue enhances an outputValue with methods specific to
// a space-independent vector quantity (e.g. averaged magnetization).
type VectorValue struct {
	outputValue
}

// NewVectorValue constructs an outputable space-independent vector quantity whose
// value is provided by function f.
func NewVectorValue(name, unit, desc string, f func() []float64) *VectorValue {
	v := &VectorValue{&valueFunc{info{3, name, unit}, f}}
	Export(v, desc)
	return v
}

func (v *VectorValue) Get() data.Vector     { return unslice(v.average()) }
func (v *VectorValue) Average() data.Vector { return v.Get() }

// outputField must be implemented by any space-dependent scalar or vector quantity
// to make it outputable.
// The space-independent counterpart is outputValue.
// Used internally by ScalarField and VectorField.
type outputField interface {
	Info
	Mesh() *data.Mesh                     // Usually the global mesh, unless this quantity has a special size.
	Slice() (q *data.Slice, recycle bool) // get quantity data (GPU or CPU), indicate need to recycle
	average() []float64
}

// NewVectorField constructs an outputable space-dependent vector quantity whose
// value is provided by function f.
func NewVectorField(name, unit, desc string, f func(dst *data.Slice)) VectorField {
	v := AsVectorField(&fieldFunc{info{3, name, unit}, f})
	Export(v, desc)
	return v
}

// NewVectorField constructs an outputable space-dependent scalar quantity whose
// value is provided by function f.
func NewScalarField(name, unit, desc string, f func(dst *data.Slice)) ScalarField {
	q := AsScalarField(&fieldFunc{info{1, name, unit}, f})
	Export(q, desc)
	return q
}

type fieldFunc struct {
	info
	f func(*data.Slice)
}

func (c *fieldFunc) Mesh() *data.Mesh   { return Mesh() }
func (c *fieldFunc) average() []float64 { return qAverageUniverse(c) }

// Calculates and returns the quantity.
// recycle is true: slice needs to be recycled.
func (q *fieldFunc) Slice() (s *data.Slice, recycle bool) {
	buf := cuda.Buffer(q.NComp(), q.Mesh().Size())
	cuda.Zero(buf)
	q.f(buf)
	return buf, true
}

// ScalarField enhances an outputField with methods specific to
// a space-dependent scalar quantity.
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

// VectorField enhances an outputField with methods specific to
// a space-dependent vector quantity.
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
