package cuda

import (
	"code.google.com/p/mx3/data"
	"code.google.com/p/mx3/kernel"
)

// Uniaxial magnetocrystalline anisotropy.
// H: anisotrotpy field in Tesla.
// M: magnetization in Tesla.
// U: anisotropy axis, length is Ku in J/m³.
func UniaxialAnisotropy(Ha, M *data.Slice, Kx, Ky, Kz float32) {

	N := Ha.Len()
	gr, bl := Make1DConf(N)

	kernel.K_uniaxialanisotropy(Ha.DevPtr(0), Ha.DevPtr(1), Ha.DevPtr(2),
		M.DevPtr(0), M.DevPtr(1), M.DevPtr(2), Kx, Ky, Kz, N, gr, bl)
}
