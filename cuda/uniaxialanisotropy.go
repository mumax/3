package cuda

import "code.google.com/p/mx3/data"

// Add uniaxial magnetocrystalline anisotropy field to Beff.
// m:  normalized magnetization.
// K:  anisotropy axis, length is in Tesla.
func AddUniaxialAnisotropy(Beff, m *data.Slice, Kx, Ky, Kz float32) {

	// TODO: size check
	N := Beff.Len()
	cfg := make1DConf(N)

	k_adduniaxialanisotropy(Beff.DevPtr(0), Beff.DevPtr(1), Beff.DevPtr(2),
		m.DevPtr(0), m.DevPtr(1), m.DevPtr(2), Kx, Ky, Kz, N, cfg)
}
