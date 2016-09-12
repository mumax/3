package cuda

import (
	"unsafe"

	"github.com/mumax/3/data"
	"github.com/mumax/3/util"
)

// Adds cubic anisotropy field to Beff.
// see cubicanisotropy.cu
func AddCubicAnisotropy(Beff, m *data.Slice, k1_red, k2_red, k3_red LUTPtr, c1, c2 LUTPtrs, regions *Bytes) {
	util.Argument(Beff.Size() == m.Size())

	N := Beff.Len()
	cfg := make1DConf(N)

	k_addcubicanisotropy_async(
		Beff.DevPtr(X), Beff.DevPtr(Y), Beff.DevPtr(Z),
		m.DevPtr(X), m.DevPtr(Y), m.DevPtr(Z),
		unsafe.Pointer(k1_red), unsafe.Pointer(k2_red), unsafe.Pointer(k3_red),
		c1[X], c1[Y], c1[Z],
		c2[X], c2[Y], c2[Z],
		regions.Ptr, N, cfg)
}

// Add uniaxial magnetocrystalline anisotropy field to Beff.
// see uniaxialanisotropy.cu
func AddUniaxialAnisotropy2(Beff, m *data.Slice, Msat, k1, k2, u MSlice) {
	util.Argument(Beff.Size() == m.Size())

	checkSize(Beff, m, k1, k2, u, Msat)

	N := Beff.Len()
	cfg := make1DConf(N)

	k_adduniaxialanisotropy2_async(
		Beff.DevPtr(X), Beff.DevPtr(Y), Beff.DevPtr(Z),
		m.DevPtr(X), m.DevPtr(Y), m.DevPtr(Z),
		Msat.DevPtr(0), Msat.Mul(0),
		k1.DevPtr(0), k1.Mul(0),
		k2.DevPtr(0), k2.Mul(0),
		u.DevPtr(X), u.Mul(X),
		u.DevPtr(Y), u.Mul(Y),
		u.DevPtr(Z), u.Mul(Z),
		N, cfg)
}
