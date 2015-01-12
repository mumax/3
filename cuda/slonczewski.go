package cuda

import (
	"unsafe"

	"github.com/mumax/3/data"
)

// Add Slonczewski ST torque to torque (Tesla).
// see slonczewski.cu
func AddSlonczewskiTorque(torque, m, J *data.Slice, fixedP LUTPtrs, Msat, alpha, pol, λ, ε_prime LUTPtr, regions *Bytes, mesh *data.Mesh) {
	N := torque.Len()
	cfg := make1DConf(N)
	thickness := float32(mesh.WorldSize()[Z])

	k_addslonczewskitorque_async(torque.DevPtr(X), torque.DevPtr(Y), torque.DevPtr(Z),
		m.DevPtr(X), m.DevPtr(Y), m.DevPtr(Z), J.DevPtr(Z),
		fixedP[X], fixedP[Y], fixedP[Z],
		unsafe.Pointer(Msat), unsafe.Pointer(alpha),
		thickness, unsafe.Pointer(pol),
		unsafe.Pointer(λ), unsafe.Pointer(ε_prime),
		regions.Ptr, N, cfg)
}
