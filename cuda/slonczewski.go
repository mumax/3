package cuda

import (
	"github.com/mumax/3/data"
)

// Add Slonczewski ST torque to torque (Tesla).
// see slonczewski2.cu
func AddSlonczewskiTorque2(torque, m *data.Slice, Msat, J, fixedP, alpha, pol, λ, ε_prime MSlice, thickness MSlice, flp float64, mesh *data.Mesh) {
	N := torque.Len()
	cfg := make1DConf(N)
	meshThickness := mesh.WorldSize()[Z]

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
		thickness.DevPtr(0), thickness.Mul(0),
		float32(meshThickness),
		float32(flp),
		N, cfg)
}
