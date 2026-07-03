package cuda

import (
	"github.com/mumax/3/data"
)

// AddVCMA adds the voltage-controlled magnetic anisotropy field to B (Tesla):
//
//	B_z = 2 * (xi * E) * m_z / (Msat * t)
//
// xi: areal VCMA coefficient (J/(V m)); E: electric field (V/m); thickness: t (m).
// Perpendicular (z) easy axis. See vcma.cu.
func AddVCMA(B, m *data.Slice, Msat, E, xi, thickness MSlice, mesh *data.Mesh) {
	N := mesh.Size()
	cfg := make3DConf(N)
	k_addvcma_async(B.DevPtr(X), B.DevPtr(Y), B.DevPtr(Z),
		m.DevPtr(X), m.DevPtr(Y), m.DevPtr(Z),
		Msat.DevPtr(0), Msat.Mul(0),
		E.DevPtr(0), E.Mul(0),
		xi.DevPtr(0), xi.Mul(0),
		thickness.DevPtr(0), thickness.Mul(0),
		N[X], N[Y], N[Z], cfg)
}
