package cuda

import (
	"code.google.com/p/mx3/data"
	"unsafe"
)

// Landau-Lifshitz torque divided by gamma0:
// 	- 1/(1+α²) [ m x B +  α m x (m x B) ]
// 	torque in Tesla
// 	m normalized
// 	B in Tesla
func LLTorque(torque, m, B *data.Slice, alpha LUTPtr, regions *Bytes) {
	N := torque.Len()
	cfg := make1DConf(N)

	k_lltorque(torque.DevPtr(0), torque.DevPtr(1), torque.DevPtr(2),
		m.DevPtr(0), m.DevPtr(1), m.DevPtr(2),
		B.DevPtr(0), B.DevPtr(1), B.DevPtr(2),
		unsafe.Pointer(alpha), regions.Ptr, N, cfg)
}
