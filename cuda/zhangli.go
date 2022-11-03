package cuda

import (
	"github.com/mumax/3/data"
)

// Add Zhang-Li ST torque (Tesla) to torque.
// see zhangli.cu
func AddZhangLiTorque(torque, m *data.Slice, Msat, J, alpha, xi, pol, g MSlice, gammaLL float32, mesh *data.Mesh) {
	c := mesh.CellSize()
	N := mesh.Size()
	cfg := make3DConf(N)

	k_addzhanglitorque2_async(
		torque.DevPtr(X), torque.DevPtr(Y), torque.DevPtr(Z),
		m.DevPtr(X), m.DevPtr(Y), m.DevPtr(Z),
		Msat.DevPtr(0), Msat.Mul(0),
		J.DevPtr(X), J.Mul(X),
		J.DevPtr(Y), J.Mul(Y),
		J.DevPtr(Z), J.Mul(Z),
		alpha.DevPtr(0), alpha.Mul(0),
		xi.DevPtr(0), xi.Mul(0),
		pol.DevPtr(0), pol.Mul(0),
		g.DevPtr(0), g.Mul(0)*gammaLL,
		float32(c[X]), float32(c[Y]), float32(c[Z]),
		N[X], N[Y], N[Z], mesh.PBC_code(), cfg)
}
