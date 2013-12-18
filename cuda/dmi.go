package cuda

import (
	"github.com/mumax/3/data"
	"github.com/mumax/3/util"
)

// Add effective field of Dzyaloshinskii-Moriya interaction to Beff (Tesla).
// According to Bagdanov and Röβler, PRL 87, 3, 2001. eq.8 (out-of-plane symmetry breaking).
// See dmi.cu
func AddDMI(Beff *data.Slice, m *data.Slice, D_redx, D_redy, D_redz, A_red float32, mesh *data.Mesh) {
	cellsize := mesh.CellSize()
	N := Beff.Size()

	util.Argument(m.Size() == Beff.Size())
	util.AssertMsg(D_redz == 0, "DMI not available along z")

	cfg := make3DConf(N)

	k_adddmi_async(Beff.DevPtr(X), Beff.DevPtr(Y), Beff.DevPtr(Z),
		m.DevPtr(X), m.DevPtr(Y), m.DevPtr(Z),
		float32(D_redx), float32(D_redy), float32(D_redz), A_red,
		float32(cellsize[X]), float32(cellsize[Y]), float32(cellsize[Z]), N[X], N[Y], N[Z], mesh.PBC_code(), cfg)
}
