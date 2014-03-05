package engine

import (
	"github.com/mumax/3/cuda"
	"github.com/mumax/3/data"
	"github.com/mumax/3/util"
	"math"
)

// Adaptive Heun method, can be used as solver.Step
func HeunStep(y *data.Slice) {
	dy0 := cuda.Buffer(VECTOR, y.Size())
	defer cuda.Recycle(dy0)

	if FixDt != 0 {
		Dt_si = FixDt
	}

	dt := float32(Dt_si * *dt_mul)
	util.Assert(dt > 0)

	// stage 1
	torqueFn(dy0)
	NEval++
	cuda.Madd2(y, y, dy0, 1, dt) // y = y + dt * dy

	// s.postStep()  // improves accuracy for good steps but painful for subtracting bad torque

	// stage 2
	dy := cuda.Buffer(3, y.Size())
	defer cuda.Recycle(dy)
	Time += Dt_si
	torqueFn(dy)
	NEval++

	// determine error
	err := 0.0
	if FixDt == 0 { // time step not fixed
		err = cuda.MaxVecDiff(dy0, dy) * float64(dt)
	}

	// adjust next time step
	if err < MaxErr || Dt_si <= MinDt { // mindt check to avoid infinite loop
		// step OK
		cuda.Madd3(y, y, dy, dy0, 1, 0.5*dt, -0.5*dt)
		solverPostStep()
		NSteps++
		adaptDt(math.Pow(MaxErr/err, 1./2.))
		LastErr = err
	} else {
		// undo bad step
		util.Assert(FixDt == 0)
		Time -= Dt_si
		cuda.Madd2(y, y, dy0, 1, -dt)
		NUndone++
		adaptDt(math.Pow(MaxErr/err, 1./3.))
	}
}
