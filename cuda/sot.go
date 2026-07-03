package cuda

import (
	"github.com/mumax/3/data"
)

// AddSOT adds the spin-orbit (spin-Hall) torque effective field to B (Tesla).
//
//	Jc:       charge current density (A/m^2) along +x
//	thetaSH:  damping-like spin-Hall angle
//	thetaFL:  field-like spin-Hall angle
//	thickness: ferromagnet thickness (m)
//
// Delivered as a compensated field so the LLG yields the DL+FL torque; spin
// polarization sigma = +y. See sot.cu.
func AddSOT(B, m *data.Slice, Msat, Jc, alpha MSlice, thetaSH, thetaFL, thickness float32, mesh *data.Mesh) {
	N := mesh.Size()
	cfg := make3DConf(N)
	k_addsot_async(B.DevPtr(X), B.DevPtr(Y), B.DevPtr(Z),
		m.DevPtr(X), m.DevPtr(Y), m.DevPtr(Z),
		Msat.DevPtr(0), Msat.Mul(0),
		Jc.DevPtr(0), Jc.Mul(0),
		alpha.DevPtr(0), alpha.Mul(0),
		thetaSH, thetaFL, thickness, N[X], N[Y], N[Z], cfg)
}
