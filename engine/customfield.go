package engine

// Add arbitrary terms to H_eff

import (
	"fmt"
	"github.com/mumax/3/cuda"
	"github.com/mumax/3/data"
	"github.com/mumax/3/util"
)

var (
	B_custom       = NewVectorField("B_custom", "T", "User-defined field", AddCustomField)
	Edens_custom   = NewScalarField("Edens_custom", "J/m3", "Energy density of user-defined field.", AddCustomEnergyDensity)
	E_custom       = NewScalarValue("E_custom", "J", "total energy of user-defined field", GetCustomEnergy)
	customTerms    []field // vector
	customEnergies []field // scalar
)

func init() {
	DeclFunc("AddFieldTerm", AddFieldTerm, "Add an expression to B_eff.")
	DeclFunc("Dot", Dot, "Dot product of two vector quantities")
	DeclFunc("Mul", Mul, "Point-wise product of two quantities")
	DeclFunc("Div", Div, "Point-wise division of two quantities")
	DeclFunc("Const", Const, "Constant, uniform number")
	DeclFunc("ConstVector", ConstVector, "Constant, uniform vector")
}

type field interface {
	Slice() (*data.Slice, bool)
	NComp() int
}

// AddFieldTerm adds a function to B_eff
func AddFieldTerm(b field) {
	customTerms = append(customTerms, b)
}

// AddCustomField evaluates the user-defined custom field terms
// and adds the result to dst.
func AddCustomField(dst *data.Slice) {
	for _, term := range customTerms {
		buf, recycle := term.Slice()
		cuda.Add(dst, dst, buf)
		if recycle {
			cuda.Recycle(buf)
		}
	}
}

// AddCustomField evaluates the user-defined custom energy density terms
// and adds the result to dst.
func AddCustomEnergyDensity(dst *data.Slice) {

}

func GetCustomEnergy() float64 {
	buf := cuda.Buffer(1, Edens_custom.Mesh().Size())
	defer cuda.Recycle(buf)
	cuda.Zero(buf)
	AddCustomEnergyDensity(buf)
	return cellVolume() * float64(cuda.Sum(buf))
}

type constValue struct {
	value []float64
}

func (c *constValue) NComp() int { return len(c.value) }

func (d *constValue) Slice() (*data.Slice, bool) {
	buf := cuda.Buffer(d.NComp(), Mesh().Size())
	for c, v := range d.value {
		cuda.Memset(buf.Comp(c), float32(v))
	}
	return buf, true
}

func Const(v float64) field {
	return &constValue{[]float64{v}}
}

func ConstVector(x, y, z float64) field {
	return &constValue{[]float64{x, y, z}}
}

// fieldOp holds the abstract functionality for operations
// (like add, multiply, ...) on space-dependend quantites
// (like M, B_sat, ...)
type fieldOp struct {
	a, b  field
	nComp int
}

func (o fieldOp) NComp() int {
	return o.nComp
}

type dotProduct struct {
	fieldOp
}

// DotProduct creates a new quantity that is the dot product of
// quantities a and b. E.g.:
// 	DotProct(&M, &B_ext)
func Dot(a, b field) field {
	return &dotProduct{fieldOp{a, b, 1}}
}

func (d *dotProduct) Slice() (*data.Slice, bool) {
	slice := cuda.Buffer(d.NComp(), Mesh().Size())
	cuda.Zero(slice)
	A, r := d.a.Slice()
	if r {
		defer cuda.Recycle(A)
	}
	B, r := d.b.Slice()
	if r {
		defer cuda.Recycle(B)
	}
	cuda.AddDotProduct(slice, 1, A, B)
	return slice, true
}

type pointwiseMul struct {
	fieldOp
}

func Mul(a, b field) field {
	nComp := -1
	switch {
	case a.NComp() == b.NComp():
		nComp = a.NComp() // vector*vector, scalar*scalar
	case a.NComp() == 1:
		nComp = b.NComp() // scalar*something
	case b.NComp() == 1:
		nComp = a.NComp() // something*scalar
	default:
		panic(fmt.Sprintf("Cannot point-wise multiply %v components by %v components", a.NComp(), b.NComp()))
	}

	return &pointwiseMul{fieldOp{a, b, nComp}}
}

func (d *pointwiseMul) Slice() (*data.Slice, bool) {
	buf := cuda.Buffer(d.NComp(), Mesh().Size())
	cuda.Zero(buf)
	a, r := d.a.Slice()
	if r {
		defer cuda.Recycle(a)
	}
	b, r := d.b.Slice()
	if r {
		defer cuda.Recycle(b)
	}

	switch {
	case a.NComp() == b.NComp():
		mulNN(buf, a, b) // vector*vector, scalar*scalar
	case a.NComp() == 1:
		mul1N(buf, a, b)
	case b.NComp() == 1:
		mul1N(buf, b, a)
	default:
		panic(fmt.Sprintf("Cannot point-wise multiply %v components by %v components", a.NComp(), b.NComp()))
	}

	return buf, true
}

// mulNN pointwise multiplies two N-component vectors,
// yielding an N-component vector stored in dst.
func mulNN(dst, a, b *data.Slice) {
	cuda.Mul(dst, a, b)
}

// mul1N pointwise multiplies a scalar (1-component) with an N-component vector,
// yielding an N-component vector stored in dst.
func mul1N(dst, a, b *data.Slice) {
	util.Assert(a.NComp() == 1)
	util.Assert(dst.NComp() == b.NComp())
	for c := 0; c < dst.NComp(); c++ {
		cuda.Mul(dst.Comp(c), a, b.Comp(c))
	}
}

type pointwiseDiv struct {
	fieldOp
}

func Div(a, b field) field {
	nComp := -1
	switch {
	case a.NComp() == b.NComp():
		nComp = a.NComp() // vector/vector, scalar/scalar
	case b.NComp() == 1:
		nComp = a.NComp() // something/scalar
	default:
		panic(fmt.Sprintf("Cannot point-wise divide %v components by %v components", a.NComp(), b.NComp()))
	}
	return &pointwiseDiv{fieldOp{a, b, nComp}}
}

func (d *pointwiseDiv) Slice() (*data.Slice, bool) {
	buf := cuda.Buffer(d.NComp(), Mesh().Size())
	a, r := d.a.Slice()
	if r {
		defer cuda.Recycle(a)
	}
	b, r := d.b.Slice()
	if r {
		defer cuda.Recycle(b)
	}

	switch {
	case a.NComp() == b.NComp():
		divNN(buf, a, b) // vector*vector, scalar*scalar
	case b.NComp() == 1:
		divN1(buf, a, b)
	default:
		panic(fmt.Sprintf("Cannot point-wise divide %v components by %v components", a.NComp(), b.NComp()))
	}

	return buf, true
}

func divNN(dst, a, b *data.Slice) {
	cuda.Div(dst, a, b)
}

func divN1(dst, a, b *data.Slice) {
	util.Assert(dst.NComp() == a.NComp())
	util.Assert(b.NComp() == 1)
	for c := 0; c < dst.NComp(); c++ {
		cuda.Div(dst.Comp(c), a.Comp(c), b)
	}
}
