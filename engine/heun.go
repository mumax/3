package engine

import (
	"github.com/mumax/3/cuda"
	"github.com/mumax/3/data"
	"github.com/mumax/3/util"
	"math"
)

// Adaptive Heun method, can be used as solver.Step
func HeunStep(s *solver, y *data.Slice) {
	dy0 := cuda.Buffer(VECTOR, y.Mesh())
	defer cuda.Recycle(dy0)

	dt := float32(s.Dt_si * s.dt_mul)
	util.Assert(dt > 0)

	// stage 1
	s.torqueFn(dy0)
	s.NEval++
	cuda.Madd2(y, y, dy0, 1, dt) // y = y + dt * dy

	// stage 2
	dy := cuda.Buffer(3, y.Mesh())
	defer cuda.Recycle(dy)
	Time += s.Dt_si
	s.torqueFn(dy)
	s.NEval++

	// determine error
	err := 0.0
	if s.FixDt == 0 { // time step not fixed
		err = cuda.MaxVecDiff(dy0, dy) * float64(dt)
	}

	// adjust next time step
	if err < s.MaxErr || s.Dt_si <= s.MinDt { // mindt check to avoid infinite loop
		// step OK
		cuda.Madd3(y, y, dy, dy0, 1, 0.5*dt, -0.5*dt)
		s.postStep(y)
		s.NSteps++
		s.adaptDt(math.Pow(s.MaxErr/err, 1./2.))
		s.LastErr = err
	} else {
		// undo bad step
		util.Assert(s.FixDt == 0)
		Time -= s.Dt_si
		cuda.Madd2(y, y, dy0, 1, -dt)
		s.NUndone++
		s.adaptDt(math.Pow(s.MaxErr/err, 1./3.))
	}
}
