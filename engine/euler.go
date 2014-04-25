package engine

import (
	"github.com/mumax/3/cuda"
	"github.com/mumax/3/util"
)

type Euler struct{}

// Euler method, can be used as solver.Step.
func (_ *Euler) Step() {
	y := M.Buffer()
	dy0 := cuda.Buffer(VECTOR, y.Size())
	defer cuda.Recycle(dy0)

	Dt_si = FixDt
	dt := float32(Dt_si * GammaLL)
	util.AssertMsg(dt > 0, "Euler solver requires fixed time step > 0")

	torqueFn(dy0)
	cuda.Madd2(y, y, dy0, 1, dt) // y = y + dt * dy
	setMaxTorque(dy0)
	setLastErr(0) // unknown

	Time += Dt_si
	M.normalize()
	NSteps++
}

func (_ *Euler) Free() {}
