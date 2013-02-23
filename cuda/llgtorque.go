package cuda

import (
	"code.google.com/p/mx3/data"
	"code.google.com/p/mx3/kernel"
)

// Calculates the REDUCED torque.
// 	- m x B +  α m x (m x B)
func CalcLLGTorque(torque, m, B *data.Slice, alpha float32) {
	N := torque.Len()
	// TODO: assert...
	gridDim, blockDim := Make1DConf(N)
	kernel.K_llgtorque(torque.DevPtr(0), torque.DevPtr(1), torque.DevPtr(2),
		m.DevPtr(0), m.DevPtr(1), m.DevPtr(2),
		B.DevPtr(0), B.DevPtr(1), B.DevPtr(2),
		alpha, N, gridDim, blockDim)
}
