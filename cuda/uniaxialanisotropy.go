package cuda

import (
	"code.google.com/p/mx3/data"
	"code.google.com/p/mx3/util"
	"unsafe"
)

// Add uniaxial magnetocrystalline anisotropy field to Beff.
func AddUniaxialAnisotropy(Beff, m *data.Slice, k1_red LUTPtr, u LUTPtrs, regions *Bytes) {
	util.Argument(Beff.Mesh().Size() == m.Mesh().Size())

	N := Beff.Len()
	cfg := make1DConf(N)

	k_adduniaxialanisotropy(Beff.DevPtr(0), Beff.DevPtr(1), Beff.DevPtr(2),
		m.DevPtr(0), m.DevPtr(1), m.DevPtr(2),
		unsafe.Pointer(k1_red), u[0], u[1], u[2],
		regions.Ptr, N, cfg)
}
