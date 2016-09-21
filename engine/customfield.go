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
	customTerms    []Q // vector
	customEnergies []Q // scalar
)

func init() {
	DeclFunc("AddFieldTerm", AddFieldTerm, "Add an expression to B_eff.")
	DeclFunc("Dot", Dot, "Dot product of two vector quantities")
	DeclFunc("Mul", Mul, "Point-wise product of two quantities")
	DeclFunc("Div", Div, "Point-wise division of two quantities")
	DeclFunc("Const", Const, "Constant, uniform number")
	DeclFunc("ConstVector", ConstVector, "Constant, uniform vector")
}

func Evalf(q Q) float64     { return 666 }
func Evalv(q Q) data.Vector { return Vector(666, 666, 666) }

// AddFieldTerm adds a function to B_eff
func AddFieldTerm(b Q) {
	customTerms = append(customTerms, b)
}

// AddCustomField evaluates the user-defined custom field terms
// and adds the result to dst.
func AddCustomField(dst *data.Slice) {
	for _, term := range customTerms {
		buf := ValueOf(term)
		cuda.Add(dst, dst, buf)
		cuda.Recycle(buf)
	}
}

// AddCustomField evaluates the user-defined custom energy density terms
// and adds the result to dst.
func AddCustomEnergyDensity(dst *data.Slice) {

}

func GetCustomEnergy() float64 {
	//buf := cuda.Buffer(1, Edens_custom.Mesh().Size())
	//defer cuda.Recycle(buf)
	//cuda.Zero(buf)
	//AddCustomEnergyDensity(buf)
	//return cellVolume() * float64(cuda.Sum(buf))
	return 999
}

type constValue struct {
	value []float64
}

func (c *constValue) NComp() int { return len(c.value) }

func (d *constValue) EvalTo(dst *data.Slice) {
	for c, v := range d.value {
		cuda.Memset(dst.Comp(c), float32(v))
	}
}

// custom MSlice implementation saves memory
func (c *constValue) MSlice() cuda.MSlice {
	return cuda.MakeMSlice(data.NilSlice(SCALAR, Mesh().Size()), c.value)
}

func Const(v float64) Q {
	return &constValue{[]float64{v}}
}

func ConstVector(x, y, z float64) Q {
	return &constValue{[]float64{x, y, z}}
}

// fieldOp holds the abstract functionality for operations
// (like add, multiply, ...) on space-dependend quantites
// (like M, B_sat, ...)
type fieldOp struct {
	a, b  Q
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
func Dot(a, b Q) Q {
	return &dotProduct{fieldOp{a, b, 1}}
}

func (d *dotProduct) EvalTo(dst *data.Slice) {
	A := ValueOf(d.a)
	defer cuda.Recycle(A)
	B := ValueOf(d.b)
	defer cuda.Recycle(B)
	cuda.Zero(dst)
	cuda.AddDotProduct(dst, 1, A, B)
}

type pointwiseMul struct {
	fieldOp
}

func Mul(a, b Q) Q {
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

func (d *pointwiseMul) EvalTo(dst *data.Slice) {
	cuda.Zero(dst)
	a := ValueOf(d.a)
	defer cuda.Recycle(a)
	b := ValueOf(d.b)
	defer cuda.Recycle(b)

	switch {
	case a.NComp() == b.NComp():
		mulNN(dst, a, b) // vector*vector, scalar*scalar
	case a.NComp() == 1:
		mul1N(dst, a, b)
	case b.NComp() == 1:
		mul1N(dst, b, a)
	default:
		panic(fmt.Sprintf("Cannot point-wise multiply %v components by %v components", a.NComp(), b.NComp()))
	}
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

func Div(a, b Q) Q {
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

func (d *pointwiseDiv) EvalTo(dst *data.Slice) {
	a := ValueOf(d.a)
	defer cuda.Recycle(a)
	b := ValueOf(d.b)
	defer cuda.Recycle(b)

	switch {
	case a.NComp() == b.NComp():
		divNN(dst, a, b) // vector*vector, scalar*scalar
	case b.NComp() == 1:
		divN1(dst, a, b)
	default:
		panic(fmt.Sprintf("Cannot point-wise divide %v components by %v components", a.NComp(), b.NComp()))
	}

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
