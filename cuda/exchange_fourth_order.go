package cuda

import (
	"unsafe"

	"github.com/mumax/3/data"
)

// Add NNN exchange field to Beff.
// 	m: normalized magnetization
// 	B: effective field in Tesla
// 	Aex_red: Aex / (Msat * 1e18 m2)
// see exchange_fourth_order.cu
func AddExchangeFourthOrder(B, m *data.Slice, AexSecond_red SymmLUT, AexFourth_red SymmLUT, Msat MSlice, regions *Bytes, mesh *data.Mesh) {
	c := mesh.CellSize()
	N := mesh.Size()
	pbc := mesh.PBC_code()
	cfg := make3DConf(N)
	k_addexchangefourthorder_async(B.DevPtr(X), B.DevPtr(Y), B.DevPtr(Z),
		m.DevPtr(X), m.DevPtr(Y), m.DevPtr(Z),
		Msat.DevPtr(0), Msat.Mul(0),
		unsafe.Pointer(AexSecond_red), unsafe.Pointer(AexFourth_red), regions.Ptr,
		float32(c[X]), float32(c[Y]), float32(c[Z]), N[X], N[Y], N[Z], pbc, cfg)
}
