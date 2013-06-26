package cuda

import (
	"code.google.com/p/mx3/data"
	"code.google.com/p/mx3/util"
	"unsafe"
)

// Add effective field of Dzyaloshinskii-Moriya interaction to Beff (Tesla).
// According to Bagdanov and Röβler, PRL 87, 3, 2001. eq.8 (out-of-plane symmetry breaking).
// m: normalized
// D_red: Tm (D / Msat)
// Beff: T
func AddDMI(Beff *data.Slice, m *data.Slice, D_red LUTPtr, regions *Bytes) {
	mesh := Beff.Mesh()
	util.Argument(m.Mesh().Size() == mesh.Size())

	N := mesh.Size()
	c := mesh.CellSize()
	cfg := make3DConf(N)

	k_adddmi(Beff.DevPtr(0), Beff.DevPtr(1), Beff.DevPtr(2),
		m.DevPtr(0), m.DevPtr(1), m.DevPtr(2),
		float32(c[0]), float32(c[1]), float32(c[2]),
		unsafe.Pointer(D_red), regions.Ptr,
		N[0], N[1], N[2], cfg)
}
