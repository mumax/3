package cuda

import (
	"github.com/mumax/3/data"
	"github.com/mumax/3/util"
	"unsafe"
)

// Adds cubic anisotropy field to Beff.
// see cubicanisotropy.cu
func AddCubicAnisotropy(Beff, m *data.Slice, k1_red LUTPtr, c1, c2 LUTPtrs, regions *Bytes) {
	util.Argument(Beff.Size() == m.Size())

	N := Beff.Len()
	cfg := make1DConf(N)

	k_addcubicanisotropy_async(
		Beff.DevPtr(X), Beff.DevPtr(Y), Beff.DevPtr(Z),
		m.DevPtr(X), m.DevPtr(Y), m.DevPtr(Z),
		unsafe.Pointer(k1_red),
		c1[X], c1[Y], c1[Z],
		c2[X], c2[Y], c2[Z],
		regions.Ptr, N, cfg)
}
