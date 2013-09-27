package engine

import (
	"github.com/mumax/3/cuda"
	"github.com/mumax/3/util"
)

// can be used as solver.step
func EulerStep(e *solver) {
	dy0 := cuda.Buffer(3, e.y.Mesh())
	defer cuda.Recycle(dy0)

	e.Dt_si = e.FixDt
	dt := float32(e.Dt_si * e.dt_mul)
	util.AssertMsg(dt > 0, "Euler solver requires fixed time step > 0")

	e.torqueFn(dy0)
	e.NEval++

	cuda.Madd2(e.y, e.y, dy0, 1, dt) // y = y + dt * dy

	Time += e.Dt_si
	e.postStep(e.y)
	e.NSteps++
}
