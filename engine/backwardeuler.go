package engine

import (
	//"fmt"
	"github.com/mumax/3/cuda"
	"github.com/mumax/3/data"
	"github.com/mumax/3/util"
	"math"
)

type BackwardEuler struct{}

// Euler method, can be used as solver.Step.
func (_ *BackwardEuler) Step() {
	util.AssertMsg(MaxErr > 0, "Backward euler solver requires MaxErr > 0")

	t0 := Time
	//Dt_si = FixDt
	dt := float32(Dt_si * GammaLL)

	y := M.Buffer()

	y0 := cuda.Buffer(VECTOR, y.Size())
	defer cuda.Recycle(y0)
	data.Copy(y0, y)

	dy := cuda.Buffer(VECTOR, y.Size())
	dy1 := cuda.Buffer(VECTOR, y.Size())
	defer cuda.Recycle(dy1)
	defer cuda.Recycle(dy)

	// Fist guess
	Time = t0 + 0.5*Dt_si // 0.5 dt makes it implicit midpoint method
	torqueFn(dy)
	cuda.Madd2(y, y0, dy, 1, dt) // y = y0 + dt * dy
	M.normalize()

	// One iteration
	torqueFn(dy1)
	cuda.Madd2(y, y0, dy1, 1, dt) // y = y0 + dt * dy1
	M.normalize()

	Time = t0 + Dt_si

	err := cuda.MaxVecDiff(dy, dy1) * float64(dt)

	// adjust next time step
	if err < MaxErr || Dt_si <= MinDt || FixDt != 0 { // mindt check to avoid infinite loop
		// step OK
		NSteps++
		adaptDt(math.Pow(MaxErr/err, 1./1.))
		setLastErr(err)
		setMaxTorque(dy)
	} else {
		// undo bad step
		util.Assert(FixDt == 0)
		Time = t0
		data.Copy(y, y0)
		NUndone++
		adaptDt(math.Pow(MaxErr/err, 1./2.))
	}

}

func (_ *BackwardEuler) Free() {}
