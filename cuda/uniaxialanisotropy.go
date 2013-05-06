package cuda

import (
	"code.google.com/p/mx3/data"
	"code.google.com/p/mx3/util"
)

// Add uniaxial magnetocrystalline anisotropy field to Beff.
// m:  normalized magnetization.
// K:  anisotropy axis in J/m³
func AddUniaxialAnisotropy(Beff, m, mask *data.Slice, Kx, Ky, Kz, Msat float64) {
	util.Argument(Beff.Mesh().Size() == m.Mesh().Size())
	if mask != nil {
		util.Argument(mask.Mesh().Size() == m.Mesh().Size())
	}

	N := Beff.Len()
	cfg := make1DConf(N)

	k_adduniaxialanisotropy(Beff.DevPtr(0), Beff.DevPtr(1), Beff.DevPtr(2),
		m.DevPtr(0), m.DevPtr(1), m.DevPtr(2),
		mask.DevPtr(0), mask.DevPtr(1), mask.DevPtr(2),
		float32(Kx/Msat), float32(Ky/Msat), float32(Kz/Msat), N, cfg)
}
