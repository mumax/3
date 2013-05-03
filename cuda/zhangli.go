package cuda

import (
	"code.google.com/p/mx3/data"
	"code.google.com/p/mx3/mag"
	"code.google.com/p/mx3/util"
)

func AddZhangLiTorque(torque, m *data.Slice, j [3]float64, Msat float64, j_MsMap *data.Slice, alpha, xi float64) {
	// TODO: assert...

	util.Argument(j_MsMap == nil) // not yet supported

	c := torque.Mesh().CellSize()
	N := torque.Mesh().Size()
	cfg := make2DConfSize(N[2], N[1], STENCIL_BLOCKSIZE)

	b := mag.MuB / (mag.Qe * Msat * (1 + xi*xi))
	ux := float32((j[0] * b) / (mag.Gamma0 * 2 * c[0]))
	uy := float32((j[1] * b) / (mag.Gamma0 * 2 * c[1]))
	uz := float32((j[2] * b) / (mag.Gamma0 * 2 * c[2]))

	k_addzhanglitorque(torque.DevPtr(0), torque.DevPtr(1), torque.DevPtr(2),
		m.DevPtr(0), m.DevPtr(1), m.DevPtr(2),
		ux, uy, uz,
		j_MsMap.DevPtr(0), j_MsMap.DevPtr(1), j_MsMap.DevPtr(2),
		float32(alpha), float32(xi),
		N[0], N[1], N[2], cfg)
}
