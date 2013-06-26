package cuda

import (
	"code.google.com/p/mx3/data"
	"code.google.com/p/mx3/mag"
	"unsafe"
)

func AddZhangLiTorque(torque, m *data.Slice, j [3]float64, bsat, alpha, xi LUTPtr, regions *Bytes) {
	// TODO: assert...

	c := torque.Mesh().CellSize()
	N := torque.Mesh().Size()
	cfg := make3DConf(N)

	ux := float32(j[0] / (mag.Gamma0 * 2 * c[0]))
	uy := float32(j[1] / (mag.Gamma0 * 2 * c[1]))
	uz := float32(j[2] / (mag.Gamma0 * 2 * c[2]))

	k_addzhanglitorque(torque.DevPtr(0), torque.DevPtr(1), torque.DevPtr(2),
		m.DevPtr(0), m.DevPtr(1), m.DevPtr(2),
		ux, uy, uz,
		unsafe.Pointer(bsat), unsafe.Pointer(alpha), unsafe.Pointer(xi), regions.Ptr,
		N[0], N[1], N[2], cfg)
}
