package engine

// Add arbitrary terms to B_eff, Edens_total.

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
	customTerms    []Quantity // vector
	customEnergies []Quantity // scalar
)

func init() {
	registerEnergy(GetCustomEnergy, AddCustomEnergyDensity)
	DeclFunc("AddFieldTerm", AddFieldTerm, "Add an expression to B_eff.")
	DeclFunc("AddEdensTerm", AddEdensTerm, "Add an expression to Edens.")
	DeclFunc("Add", Add, "Add two quantities")
	DeclFunc("Madd", Madd, "Weighted addition: Madd(Q1,Q2,c1,c2) = c1*Q1 + c2*Q2")
	DeclFunc("Dot", Dot, "Dot product of two vector quantities")
	DeclFunc("Cross", Cross, "Cross product of two vector quantities")
	DeclFunc("Mul", Mul, "Point-wise product of two quantities")
	DeclFunc("MulMV", MulMV, "Matrix-Vector product: MulMV(AX, AY, AZ, m) = (AX·m, AY·m, AZ·m). "+
		"The arguments Ax, Ay, Az and m are quantities with 3 componets.")
	DeclFunc("Div", Div, "Point-wise division of two quantities")
	DeclFunc("Const", Const, "Constant, uniform number")
	DeclFunc("ConstVector", ConstVector, "Constant, uniform vector")
	DeclFunc("Shifted", Shifted, "Shifted quantity")
	DeclFunc("Masked", Masked, "Mask quantity with shape")
	DeclFunc("RemoveCustomFields", RemoveCustomFields, "Removes all custom fields again")
}

//Removes all customfields
func RemoveCustomFields() {
	customTerms = nil
}

// AddFieldTerm adds an effective field function (returning Teslas) to B_eff.
// Be sure to also add the corresponding energy term using AddEnergyTerm.
func AddFieldTerm(b Quantity) {
	customTerms = append(customTerms, b)
}

// AddEnergyTerm adds an energy density function (returning Joules/m³) to Edens_total.
// Needed when AddFieldTerm was used and a correct energy is needed
// (e.g. for Relax, Minimize, ...).
func AddEdensTerm(e Quantity) {
	customEnergies = append(customEnergies, e)
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

// Adds the custom energy densities (defined with AddCustomE
func AddCustomEnergyDensity(dst *data.Slice) {
	for _, term := range customEnergies {
		buf := ValueOf(term)
		cuda.Add(dst, dst, buf)
		cuda.Recycle(buf)
	}
}

func GetCustomEnergy() float64 {
	buf := cuda.Buffer(1, Mesh().Size())
	defer cuda.Recycle(buf)
	cuda.Zero(buf)
	AddCustomEnergyDensity(buf)
	return cellVolume() * float64(cuda.Sum(buf))
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

// Const returns a constant (uniform) scalar quantity,
// that can be used to construct custom field terms.
func Const(v float64) Quantity {
	return &constValue{[]float64{v}}
}

// ConstVector returns a constant (uniform) vector quantity,
// that can be used to construct custom field terms.
func ConstVector(x, y, z float64) Quantity {
	return &constValue{[]float64{x, y, z}}
}

// fieldOp holds the abstract functionality for operations
// (like add, multiply, ...) on space-dependend quantites
// (like M, B_sat, ...)
type fieldOp struct {
	a, b  Quantity
	nComp int
}

func (o fieldOp) NComp() int {
	return o.nComp
}

type dotProduct struct {
	fieldOp
}

type crossProduct struct {
	fieldOp
}

type addition struct {
	fieldOp
}

type mAddition struct {
	fieldOp
	fac1, fac2 float64
}

type mulmv struct {
	ax, ay, az, b Quantity
}

// MulMV returns a new Quantity that evaluates to the
// matrix-vector product (Ax·b, Ay·b, Az·b).
func MulMV(Ax, Ay, Az, b Quantity) Quantity {
	util.Argument(Ax.NComp() == 3 &&
		Ay.NComp() == 3 &&
		Az.NComp() == 3 &&
		b.NComp() == 3)
	return &mulmv{Ax, Ay, Az, b}
}

func (q *mulmv) EvalTo(dst *data.Slice) {
	util.Argument(dst.NComp() == 3)
	cuda.Zero(dst)
	b := ValueOf(q.b)
	defer cuda.Recycle(b)

	{
		Ax := ValueOf(q.ax)
		cuda.AddDotProduct(dst.Comp(X), 1, Ax, b)
		cuda.Recycle(Ax)
	}
	{

		Ay := ValueOf(q.ay)
		cuda.AddDotProduct(dst.Comp(Y), 1, Ay, b)
		cuda.Recycle(Ay)
	}
	{
		Az := ValueOf(q.az)
		cuda.AddDotProduct(dst.Comp(Z), 1, Az, b)
		cuda.Recycle(Az)
	}
}

func (q *mulmv) NComp() int {
	return 3
}

// DotProduct creates a new quantity that is the dot product of
// quantities a and b. E.g.:
// 	DotProct(&M, &B_ext)
func Dot(a, b Quantity) Quantity {
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

// CrossProduct creates a new quantity that is the cross product of
// quantities a and b. E.g.:
// 	CrossProct(&M, &B_ext)
func Cross(a, b Quantity) Quantity {
	return &crossProduct{fieldOp{a, b, 3}}
}

func (d *crossProduct) EvalTo(dst *data.Slice) {
	A := ValueOf(d.a)
	defer cuda.Recycle(A)
	B := ValueOf(d.b)
	defer cuda.Recycle(B)
	cuda.Zero(dst)
	cuda.CrossProduct(dst, A, B)
}

func Add(a, b Quantity) Quantity {
	if a.NComp() != b.NComp() {
		panic(fmt.Sprintf("Cannot point-wise Add %v components by %v components", a.NComp(), b.NComp()))
	}
	return &addition{fieldOp{a, b, a.NComp()}}
}

func (d *addition) EvalTo(dst *data.Slice) {
	A := ValueOf(d.a)
	defer cuda.Recycle(A)
	B := ValueOf(d.b)
	defer cuda.Recycle(B)
	cuda.Zero(dst)
	cuda.Add(dst, A, B)
}

type pointwiseMul struct {
	fieldOp
}

func Madd(a, b Quantity, fac1, fac2 float64) *mAddition {
	if a.NComp() != b.NComp() {
		panic(fmt.Sprintf("Cannot point-wise add %v components by %v components", a.NComp(), b.NComp()))
	}
	return &mAddition{fieldOp{a, b, a.NComp()}, fac1, fac2}
}

func (o *mAddition) EvalTo(dst *data.Slice) {
	A := ValueOf(o.a)
	defer cuda.Recycle(A)
	B := ValueOf(o.b)
	defer cuda.Recycle(B)
	cuda.Zero(dst)
	cuda.Madd2(dst, A, B, float32(o.fac1), float32(o.fac2))
}

// Mul returns a new quantity that evaluates to the pointwise product a and b.
func Mul(a, b Quantity) Quantity {
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

// Div returns a new quantity that evaluates to the pointwise product a and b.
func Div(a, b Quantity) Quantity {
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

type shifted struct {
	orig       Quantity
	dx, dy, dz int
}

// Shifted returns a new Quantity that evaluates to
// the original, shifted over dx, dy, dz cells.
func Shifted(q Quantity, dx, dy, dz int) Quantity {
	util.Assert(dx != 0 || dy != 0 || dz != 0)
	return &shifted{q, dx, dy, dz}
}

func (q *shifted) EvalTo(dst *data.Slice) {
	orig := ValueOf(q.orig)
	defer cuda.Recycle(orig)
	for i := 0; i < q.NComp(); i++ {
		dsti := dst.Comp(i)
		origi := orig.Comp(i)
		if q.dx != 0 {
			cuda.ShiftX(dsti, origi, q.dx, 0, 0)
		}
		if q.dy != 0 {
			cuda.ShiftY(dsti, origi, q.dy, 0, 0)
		}
		if q.dz != 0 {
			cuda.ShiftZ(dsti, origi, q.dz, 0, 0)
		}
	}
}

func (q *shifted) NComp() int {
	return q.orig.NComp()
}

// Masks a quantity with a shape
// The shape will be only evaluated once on the mesh,
// and will be re-evaluated after mesh change,
// because otherwise too slow
func Masked(q Quantity, shape Shape) Quantity {
	return &masked{q, shape, nil, data.Mesh{}}
}

type masked struct {
	orig  Quantity
	shape Shape
	mask  *data.Slice
	mesh  data.Mesh
}

func (q *masked) EvalTo(dst *data.Slice) {
	if q.mesh != *Mesh() {
		// When mesh is changed, mask needs an update
		q.createMask()
	}
	orig := ValueOf(q.orig)
	defer cuda.Recycle(orig)
	mul1N(dst, q.mask, orig)
}

func (q *masked) NComp() int {
	return q.orig.NComp()
}

func (q *masked) createMask() {
	size := Mesh().Size()
	// Prepare mask on host
	maskhost := data.NewSlice(SCALAR, size)
	defer maskhost.Free()
	maskScalars := maskhost.Scalars()
	for iz := 0; iz < size[Z]; iz++ {
		for iy := 0; iy < size[Y]; iy++ {
			for ix := 0; ix < size[X]; ix++ {
				r := Index2Coord(ix, iy, iz)
				if q.shape(r[X], r[Y], r[Z]) {
					maskScalars[iz][iy][ix] = 1
				}
			}
		}
	}
	// Update mask
	q.mask.Free()
	q.mask = cuda.NewSlice(SCALAR, size)
	data.Copy(q.mask, maskhost)
	q.mesh = *Mesh()
	// Remove mask from host
}
