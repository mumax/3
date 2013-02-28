package cuda

import (
	"code.google.com/p/mx3/data"
	"code.google.com/p/mx3/kernel"
)

// Landau-Lifshitz torque divided by gamma0:
// 	- 1/(1+α²) [ M x B +  α (M/|M|) x (M x B) ]
// 	torque in Tesla/s
// 	M in Tesla
// 	B in Tesla
func LLGTorque(torque, m, B *data.Slice, alpha float32) {
	// TODO: assert...

	N := torque.Len()
	gr, bl := Make1DConf(N)

	kernel.K_llgtorque(torque.DevPtr(0), torque.DevPtr(1), torque.DevPtr(2),
		m.DevPtr(0), m.DevPtr(1), m.DevPtr(2),
		B.DevPtr(0), B.DevPtr(1), B.DevPtr(2),
		alpha, N, gr, bl)
}

// Only the damping term of LLGTorque, with alpha 1. Useful for relaxation.
func DampingTorque(torque, m, B *data.Slice) {
	N := torque.Len()
	gr, bl := Make1DConf(N)

	kernel.K_dampingtorque(torque.DevPtr(0), torque.DevPtr(1), torque.DevPtr(2),
		m.DevPtr(0), m.DevPtr(1), m.DevPtr(2), B.DevPtr(0), B.DevPtr(1), B.DevPtr(2), N, gr, bl)
}
