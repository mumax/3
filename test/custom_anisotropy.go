/*
	Test custom field implementation.
	Like uniaxialanisotropy.mx3, but with custom anisotropy implementation.
*/

package main

import (
	. "github.com/mumax/3/engine"
)

func main() {

	defer InitAndClose()()

	SetGridSize(64, 64, 1)
	SetCellSize(4e-9, 4e-9, 2e-9)

	Aex.Set(13e-12)
	Alpha.Set(1)
	M.Set(Uniform(1, 1, 0))

	Msat.Set(1100e3)
	K := 0.5e6
	u := ConstVector(1, 0, 0)

	prefactor := Const((2 * K) / (Msat.Average()))
	MyAnis := Mul(prefactor, Mul(Dot(u, &M), u))
	AddFieldTerm(MyAnis)

	B_ext.Set(Vector(0, 0.00, 0))
	Relax()
	Expect("my", M.Average()[1], 0.000, 1e-3)

	B_ext.Set(Vector(0, 0.01, 0))
	Relax()
	Expect("my", M.Average()[1], 0.011, 1e-3)

	B_ext.Set(Vector(0, 0.03, 0))
	Relax()
	Expect("my", M.Average()[1], 0.033, 1e-3)

	B_ext.Set(Vector(0, 0.10, 0))
	Relax()
	Expect("my", M.Average()[1], 0.110, 1e-3)

	B_ext.Set(Vector(0, 0.30, 0))
	Relax()
	Expect("my", M.Average()[1], 0.331, 1e-3)

}
