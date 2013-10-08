package cuda

import (
	"github.com/mumax/3/data"
	"unsafe"
)

func AddZhangLiTorque(torque, m, jpol *data.Slice, bsat, alpha, xi, pol LUTPtr, regions *Bytes) {

	mesh := torque.Mesh()

	c := mesh.CellSize()
	N := mesh.Size()
	cfg := make3DConf(N)

	k_addzhanglitorque(torque.DevPtr(0), torque.DevPtr(1), torque.DevPtr(2),
		m.DevPtr(0), m.DevPtr(1), m.DevPtr(2),
		jpol.DevPtr(0), jpol.DevPtr(1), jpol.DevPtr(2),
		float32(c[0]), float32(c[1]), float32(c[2]),
		unsafe.Pointer(bsat), unsafe.Pointer(alpha), unsafe.Pointer(xi), unsafe.Pointer(pol),
		regions.Ptr, N[0], N[1], N[2], mesh.PBC_code(), cfg)
}
