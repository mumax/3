package cuda

import (
	"github.com/mumax/3/v3/data"
)

// m = 1 / (4 + τ²(m x H)²) [{4 - τ²(m x H)²} m - 4τ(m x m x H)]
// note: torque from LLNoPrecess has negative sign
func Minimize(m, m0, torque *data.Slice, dt float32) {
	N := m.Len()
	cfg := make1DConf(N)

	k_minimize_async(m.DevPtr(X), m.DevPtr(Y), m.DevPtr(Z),
		m0.DevPtr(X), m0.DevPtr(Y), m0.DevPtr(Z),
		torque.DevPtr(X), torque.DevPtr(Y), torque.DevPtr(Z),
		dt, N, cfg)
}
