package cuda

import (
	"github.com/mumax/3/data"
	"github.com/mumax/3/util"
	"unsafe"
)

func AddCubicAnisotropy(Beff, m *data.Slice, k1_red LUTPtr, c1, c2 LUTPtrs, regions *Bytes) {
	util.Argument(Beff.Mesh().Size() == m.Mesh().Size())

	N := Beff.Len()
	cfg := make1DConf(N)

	k_addcubicanisotropy(
		Beff.DevPtr(0), Beff.DevPtr(1), Beff.DevPtr(2),
		m.DevPtr(0), m.DevPtr(1), m.DevPtr(2),
		unsafe.Pointer(k1_red),
		c1[0], c1[1], c1[2],
		c2[0], c2[1], c2[2],
		regions.Ptr, N, cfg)
}
