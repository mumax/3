package cuda

import (
	"code.google.com/p/mx3/data"
	. "code.google.com/p/mx3/mag"
	"code.google.com/p/mx3/util"
)

// be careful with gamma!
//P := s.spinPol put in j
func AddZhangLiTorque(torque, m *data.Slice, j [3]float64, Msat float64, j_MsMap *data.Slice, alpha, xi float64) {
	// TODO: assert...

	util.Argument(j_MsMap == nil) // not yet supported

	cfg := make1DConf(torque.Len())
	s := torque.Mesh().Size()
	c := torque.Mesh().CellSize()

	b := MuB / (Qe * Msat * (1 + xi*xi))
	ux := float32((j[0] * b) / (2 * c[0]))
	uy := float32((j[1] * b) / (2 * c[1]))
	uz := float32((j[2] * b) / (2 * c[2]))

	k_addzhanglitorque(torque.DevPtr(0), torque.DevPtr(1), torque.DevPtr(2),
		m.DevPtr(0), m.DevPtr(1), m.DevPtr(2),
		ux, uy, uz,
		j_MsMap.DevPtr(0), j_MsMap.DevPtr(1), j_MsMap.DevPtr(2),
		float32(alpha), float32(xi),
		s[0], s[1], s[2], cfg)
}
