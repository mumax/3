package engine

import (
	"github.com/mumax/3/v3/cuda"
	"github.com/mumax/3/v3/util"
)

type Euler struct{}

// Euler method, can be used as solver.Step.
func (_ *Euler) Step() {
	y := M.Buffer()
	dy0 := cuda.Buffer(VECTOR, y.Size())
	defer cuda.Recycle(dy0)

	torqueFn(dy0)
	setMaxTorque(dy0)

	// Adaptive time stepping: treat MaxErr as the maximum magnetization delta
	// (proportional to the error, but an overestimation for sure)
	var dt float32
	if FixDt != 0 {
		Dt_si = FixDt
		dt = float32(Dt_si * GammaLL)
	} else {
		dt = float32(MaxErr / LastTorque)
		Dt_si = float64(dt) / GammaLL
	}
	util.AssertMsg(dt > 0, "Euler solver requires fixed time step > 0")
	setLastErr(float64(dt) * LastTorque)

	cuda.Madd2(y, y, dy0, 1, dt) // y = y + dt * dy
	M.normalize()
	Time += Dt_si
	NSteps++
}

func (_ *Euler) Free() {}
