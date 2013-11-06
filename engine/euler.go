package engine

import (
	"github.com/mumax/3/cuda"
	"github.com/mumax/3/util"
)

// Euler method, can be used as solver.Step.
func EulerStep(s *solver) {
	dy0 := cuda.Buffer(VECTOR, s.y.Mesh())
	defer cuda.Recycle(dy0)

	s.Dt_si = s.FixDt
	dt := float32(s.Dt_si * s.dt_mul)
	util.AssertMsg(dt > 0, "Euler solver requires fixed time step > 0")

	s.torqueFn(dy0)
	s.NEval++

	cuda.Madd2(s.y, s.y, dy0, 1, dt) // y = y + dt * dy

	Time += s.Dt_si
	s.postStep(s.y)
	s.NSteps++
}
