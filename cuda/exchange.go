package cuda

import (
	"code.google.com/p/mx3/data"
	"unsafe"
)

//#include "stencil.h"
import "C"

const STENCIL_BLOCKSIZE = C.STENCIL_BLOCKSIZE_X

// Add exchange field to Beff.
// 	m: normalized magnetization
// 	B: effective field in Tesla
// 	Aex_red: 2*Aex / (Msat * 1e18 m2)
func AddExchange(B, m *data.Slice, Aex_red SymmLUT, regions *Bytes) {
	mesh := B.Mesh()
	c := mesh.CellSize()
	w0 := float32(1e-18 / (c[0] * c[0]))
	w1 := float32(1e-18 / (c[1] * c[1]))
	w2 := float32(1e-18 / (c[2] * c[2]))
	N := mesh.Size()
	cfg := make2DConfSize(N[1], N[2], STENCIL_BLOCKSIZE)
	k_addexchange(B.DevPtr(0), B.DevPtr(1), B.DevPtr(2),
		m.DevPtr(0), m.DevPtr(1), m.DevPtr(2),
		unsafe.Pointer(Aex_red), regions.Ptr,
		w0, w1, w2, N[0], N[1], N[2], cfg)
}
