package cuda

import (
	"code.google.com/p/mx3/data"
	"code.google.com/p/mx3/mag"
)

// Add uniaxial magnetocrystalline anisotropy field to Beff.
// m:  normalized magnetization.
// U:  anisotropy axis, length is Ku/Bsat in J/Tm³.
func AddUniaxialAnisotropy(Beff, m *data.Slice, Kx, Ky, Kz float32) {

	// TODO: size check
	N := Beff.Len()
	const µ0 = mag.Mu0
	cfg := make1DConf(N)

	k_adduniaxialanisotropy(Beff.DevPtr(0), Beff.DevPtr(1), Beff.DevPtr(2),
		m.DevPtr(0), m.DevPtr(1), m.DevPtr(2), µ0*Kx, µ0*Ky, µ0*Kz, N, cfg)
}
