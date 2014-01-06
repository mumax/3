package cuda

import (
	"github.com/mumax/3/data"
	"unsafe"
)

// Add exchange field to Beff.
// 	m: normalized magnetization
// 	B: effective field in Tesla
// 	Aex_red: 2*Aex / (Msat * 1e18 m2)
// see exchange.cu
func AddExchange(B, m *data.Slice, Aex_red SymmLUT, regions *Bytes, mesh *data.Mesh) {
	c := mesh.CellSize()
	wx := float32(1e-18 / (c[X] * c[X]))
	wy := float32(1e-18 / (c[Y] * c[Y]))
	wz := float32(1e-18 / (c[Z] * c[Z]))
	N := mesh.Size()
	pbc := mesh.PBC_code()
	cfg := make3DConf(N)
	k_addexchange_async(B.DevPtr(X), B.DevPtr(Y), B.DevPtr(Z),
		m.DevPtr(X), m.DevPtr(Y), m.DevPtr(Z),
		unsafe.Pointer(Aex_red), regions.Ptr,
		wx, wy, wz, N[X], N[Y], N[Z], pbc, cfg)
}
