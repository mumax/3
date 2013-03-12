package cuda

import (
	"code.google.com/p/mx3/data"
	"code.google.com/p/mx3/mag"
)

// Uniaxial magnetocrystalline anisotropy.
// Ba: anisotrotpy field in Tesla.
// M:  magnetization in Tesla.
// U:  anisotropy axis, length is Ku in J/m³.
func UniaxialAnisotropy(Ba, M *data.Slice, Kx, Ky, Kz float32) {

	N := Ba.Len()
	const µ0 = mag.Mu0
	cfg := make1DConf(N)

	k_uniaxialanisotropy(Ba.DevPtr(0), Ba.DevPtr(1), Ba.DevPtr(2),
		M.DevPtr(0), M.DevPtr(1), M.DevPtr(2), µ0*Kx, µ0*Ky, µ0*Kz, N, cfg)
}
