package cuda

import (
	"github.com/mumax/3/data"
	"github.com/mumax/3/util"
)

// Add effective field of Dzyaloshinskii-Moriya interaction to Beff (Tesla).
// According to Bagdanov and Röβler, PRL 87, 3, 2001. eq.8 (out-of-plane symmetry breaking).
// See dmi.cu
func AddDMI(Beff *data.Slice, m *data.Slice, D_redx, D_redy, D_redz, A_red float32, str int) {
	mesh := Beff.Mesh()
	N := mesh.Size()
	c := mesh.CellSize()

	util.Argument(m.Mesh().Size() == mesh.Size())
	util.AssertMsg(N[Z] == 1, "DMI available in 2D only")
	util.AssertMsg(mesh.PBC_code() == 0, "DMI not available with PBC")
	util.AssertMsg(D_redz == 0, "not available along z")

	cfg := make3DConf(N)

	k_adddmi_async(Beff.DevPtr(X), Beff.DevPtr(Y), Beff.DevPtr(Z),
		m.DevPtr(X), m.DevPtr(Y), m.DevPtr(Z),
		float32(D_redx), float32(D_redy), float32(D_redz), A_red,
		float32(c[X]), float32(c[Y]), float32(c[Z]), N[X], N[Y], N[Z], cfg, str)
}
