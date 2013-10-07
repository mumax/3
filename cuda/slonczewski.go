package cuda

import (
	"github.com/mumax/3/data"
	"unsafe"
)

func AddSlonczewskiTorque(torque, m, J *data.Slice, fixedP LUTPtrs, Msat, alpha, pol, λ, ε_prime LUTPtr, regions *Bytes) {

	N := torque.Len()
	cfg := make1DConf(N)
	thickness := float32(torque.Mesh().CellSize()[0])

	k_addslonczewskitorque(torque.DevPtr(0), torque.DevPtr(1), torque.DevPtr(2),
		m.DevPtr(0), m.DevPtr(1), m.DevPtr(2),
		J.DevPtr(0), J.DevPtr(1), J.DevPtr(2),
		fixedP[0], fixedP[1], fixedP[2],
		unsafe.Pointer(Msat), unsafe.Pointer(alpha),
		thickness, unsafe.Pointer(pol),
		unsafe.Pointer(λ), unsafe.Pointer(ε_prime),
		regions.Ptr, N, cfg)
}
