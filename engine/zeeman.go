package engine

import (
	"github.com/mumax/3/cuda"
	"github.com/mumax/3/data"
	"github.com/mumax/3/mag"
)

var (
	B_ext        excitation
	E_Zeeman     *GetScalar
	Edens_zeeman adder
)

func init() {
	B_ext.init("B_ext", "T", "Externally applied field")
	E_Zeeman = NewGetScalar("E_Zeeman", "J", "Zeeman energy", GetZeemanEnergy)
	Edens_zeeman.init(SCALAR, "Edens_Zeeman", "J/m3", "Zeeman energy density", AddZeemanEdens)
	registerEnergy(GetZeemanEnergy, AddZeemanEdens)
}

func GetZeemanEnergy() float64 {
	return -1 * cellVolume() * dot(&M_full, &B_ext)
}

func AddZeemanEdens(dst *data.Slice) {
	B, r := B_ext.Slice()
	if r {
		defer cuda.Recycle(B)
	}
	prefactor := float32(-mag.Mu0)
	cuda.AddDotProduct(dst, prefactor, B, M.Buffer(), geometry.Gpu())
}
