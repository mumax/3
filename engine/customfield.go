package engine

// Add arbitrary terms to H_eff

import (
	"github.com/mumax/3/cuda"
	"github.com/mumax/3/data"
)

var (
	B_custom       = NewVectorField("B_custom", "T", "User-defined field", AddCustomField)
	Edens_custom   = NewScalarField("Edens_custom", "J/m3", "Energy density of user-defined field.", AddCustomEnergyDensity)
	E_custom       = NewScalarValue("E_custom", "J", "total energy of user-defined field", GetCustomEnergy)
	customTerms    []VectorField
	customEnergies []ScalarField
)

func init() {
	DeclFunc("AddFieldTerm", AddFieldTerm, "Add an expression to B_eff.")
	DeclFunc("Dot", Dot, "Dot product of two vector quantities")
	DeclFunc("Mul", Mul, "Point-wise product of two quantities")
	DeclFunc("Div", Div, "Point-wise division of two quantities")
}

func AddFieldTerm(b outputField) {
	customTerms = append(customTerms, AsVectorField(b))
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

// fieldOp holds the abstract functionality for operations
// (like add, multiply, ...) on space-dependend quantites
// (like M, B_sat, ...)
type fieldOp struct {
	a, b outputField
	*info
}

func (o fieldOp) Mesh() *data.Mesh {
	return o.a.Mesh()
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
func Dot(a, b outputField) outputField {
	return &dotProduct{fieldOp{a, b, &info{
		nComp: 1,
		name:  a.Name() + "_dot_" + b.Name(),
		unit:  a.Unit() + "*" + b.Unit()}}}
}

func (d *dotProduct) Slice() (*data.Slice, bool) {
	slice := cuda.Buffer(d.NComp(), d.Mesh().Size())
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

func (d *dotProduct) average() []float64 {
	return qAverageUniverse(d)
}

func (d *dotProduct) Average() float64 {
	return d.average()[0]
}

type pointwiseMul struct {
	fieldOp
}

func Mul(a, b outputField) outputField {
	return &pointwiseMul{fieldOp{a, b, &info{
		nComp: a.NComp(),
		name:  a.Name() + "_mul_" + b.Name(),
		unit:  a.Unit() + "*" + b.Unit()}}}
}

func (d *pointwiseMul) Slice() (*data.Slice, bool) {
	slice := cuda.Buffer(d.NComp(), d.Mesh().Size())
	cuda.Zero(slice)
	A, r := d.a.Slice()
	if r {
		defer cuda.Recycle(A)
	}
	B, r := d.b.Slice()
	if r {
		defer cuda.Recycle(B)
	}
	cuda.Mul(slice, A, B)
	return slice, true
}

func (d *pointwiseMul) average() []float64 {
	return qAverageUniverse(d)
}


type pointwiseDiv struct {
	fieldOp
}

func Div(a, b outputField) outputField {
	return &pointwiseDiv{fieldOp{a, b, &info{
		nComp: a.NComp(),
		name:  a.Name() + "_mul_" + b.Name(),
		unit:  a.Unit() + "*" + b.Unit()}}}
}

func (d *pointwiseDiv) Slice() (*data.Slice, bool) {
	slice := cuda.Buffer(d.NComp(), d.Mesh().Size())
	cuda.Zero(slice)
	A, r := d.a.Slice()
	if r {
		defer cuda.Recycle(A)
	}
	B, r := d.b.Slice()
	if r {
		defer cuda.Recycle(B)
	}
	cuda.Div(slice, A, B)
	return slice, true
}

func (d *pointwiseDiv) average() []float64 {
	return qAverageUniverse(d)
}
