package cuda

import (
	"unsafe"

	"github.com/mumax/3/data"
)

// Add Slonczewski ST torque to torque (Tesla).
// see slonczewski.cu
func AddSlonczewskiTorque(torque, m, J, fixedP *data.Slice, Msat, alpha, pol, λ, ε_prime LUTPtr, regions *Bytes, mesh *data.Mesh) {
	N := torque.Len()
	cfg := make1DConf(N)
	thickness := float32(mesh.WorldSize()[Z])

	k_addslonczewskitorque_async(torque.DevPtr(X), torque.DevPtr(Y), torque.DevPtr(Z),
		m.DevPtr(X), m.DevPtr(Y), m.DevPtr(Z), J.DevPtr(Z),
		fixedP.DevPtr(X), fixedP.DevPtr(Y), fixedP.DevPtr(Z),
		unsafe.Pointer(Msat), unsafe.Pointer(alpha),
		thickness, unsafe.Pointer(pol),
		unsafe.Pointer(λ), unsafe.Pointer(ε_prime),
		regions.Ptr, N, cfg)
}

// Add Slonczewski ST torque to torque (Tesla).
// see slonczewski.cu
func AddSlonczewskiTorque2(torque, m *data.Slice, Msat, J, fixedP, alpha, pol, λ, ε_prime MSlice, mesh *data.Mesh) {
	N := torque.Len()
	cfg := make1DConf(N)
	flt := float32(mesh.WorldSize()[Z])

	k_addslonczewskitorque2_async(
		torque.DevPtr(X), torque.DevPtr(Y), torque.DevPtr(Z),
		m.DevPtr(X), m.DevPtr(Y), m.DevPtr(Z),
		Msat.DevPtr(0), Msat.Mul(0),
		J.DevPtr(Z), J.Mul(Z),
		fixedP.DevPtr(X), fixedP.Mul(X),
		fixedP.DevPtr(Y), fixedP.Mul(Y),
		fixedP.DevPtr(Z), fixedP.Mul(Z),
		alpha.DevPtr(0), alpha.Mul(0),
		pol.DevPtr(0), pol.Mul(0),
		λ.DevPtr(0), λ.Mul(0),
		ε_prime.DevPtr(0), ε_prime.Mul(0),
		unsafe.Pointer(uintptr(0)), flt,
		N, cfg)
}
