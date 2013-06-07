package cuda

import (
	"code.google.com/p/mx3/data"
	"code.google.com/p/mx3/util"
)

// Add uniaxial magnetocrystalline anisotropy field to Beff.
// m:  normalized magnetization.
// TODO: doc
func AddUniaxialAnisotropy(Beff, m *data.Slice, ku1_red LUTs, regions *Bytes) {
	util.Argument(Beff.Mesh().Size() == m.Mesh().Size())

	N := Beff.Len()
	cfg := make1DConf(N)

	k_adduniaxialanisotropy(Beff.DevPtr(0), Beff.DevPtr(1), Beff.DevPtr(2),
		m.DevPtr(0), m.DevPtr(1), m.DevPtr(2),
		ku1_red.Ptr(0), ku1_red.Ptr(1), ku1_red.Ptr(2),
		regions.Ptr, N, cfg)
}
