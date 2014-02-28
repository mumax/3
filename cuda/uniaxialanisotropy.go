package cuda

import (
	"github.com/mumax/3/data"
	"github.com/mumax/3/util"
	"unsafe"
)

// Add uniaxial magnetocrystalline anisotropy field to Beff.
// see uniaxialanisotropy.cu
func AddUniaxialAnisotropy(Beff, m *data.Slice, k1_red data.LUTPtr, u data.LUTPtrs, regions *Bytes) {
	util.Argument(Beff.Size() == m.Size())

	N := Beff.Len()
	cfg := make1DConf(N)

	k_adduniaxialanisotropy_async(Beff.DevPtr(X), Beff.DevPtr(Y), Beff.DevPtr(Z),
		m.DevPtr(X), m.DevPtr(Y), m.DevPtr(Z),
		unsafe.Pointer(k1_red), u[X], u[Y], u[Z],
		regions.Ptr, N, cfg)
}
