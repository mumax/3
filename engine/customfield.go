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

func init() {
	DeclFunc("Dot", Dot, "Dot product of two vector quantities")
}

type dotProduct struct {
	a, b outputField
}

// DotProduct creates a new quantity that is the dot product of
// quantities a and b. E.g.:
// 	DotProct(&M, &B_ext)
func Dot(a, b outputField) *dotProduct {
	return &dotProduct{a, b}
}

func (d *dotProduct) Mesh() *data.Mesh {
	return d.a.Mesh()
}

func (d *dotProduct) NComp() int {
	return 1
}

func (d *dotProduct) Name() string {
	return d.a.Name() + "_dot_" + d.b.Name()
}

func (d *dotProduct) Unit() string {
	return d.a.Unit() + d.b.Unit()
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
