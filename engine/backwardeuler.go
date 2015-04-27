package engine

import (
	//"fmt"
	"github.com/mumax/3/cuda"
	"github.com/mumax/3/data"
	"github.com/mumax/3/util"
)

type BackwardEuler struct {
	dy1 *data.Slice
}

// Euler method, can be used as solver.Step.
func (s *BackwardEuler) Step() {
	util.AssertMsg(MaxErr > 0, "Backward euler solver requires MaxErr > 0")

	t0 := Time

	y := M.Buffer()

	y0 := cuda.Buffer(VECTOR, y.Size())
	defer cuda.Recycle(y0)
	data.Copy(y0, y)

	dy0 := cuda.Buffer(VECTOR, y.Size())
	defer cuda.Recycle(dy0)
	if s.dy1 == nil {
		s.dy1 = cuda.Buffer(VECTOR, y.Size())
	}
	dy1 := s.dy1

	Dt_si = FixDt
	dt := float32(Dt_si * GammaLL)
	util.AssertMsg(dt > 0, "Backward Euler solver requires fixed time step > 0")

	// Fist guess
	Time = t0 + 0.5*Dt_si // 0.5 dt makes it implicit midpoint method

	// with temperature, previous torque cannot be used as predictor
	if Temp.isZero() {
		cuda.Madd2(y, y0, dy1, 1, dt) // predictor euler step with previous torque
		M.normalize()
	}

	torqueFn(dy0)
	cuda.Madd2(y, y0, dy0, 1, dt) // y = y0 + dt * dy
	M.normalize()

	// One iteration
	torqueFn(dy1)
	cuda.Madd2(y, y0, dy1, 1, dt) // y = y0 + dt * dy1
	M.normalize()

	Time = t0 + Dt_si

	err := cuda.MaxVecDiff(dy0, dy1) * float64(dt)

	// adjust next time step
	//if err < MaxErr || Dt_si <= MinDt || FixDt != 0 { // mindt check to avoid infinite loop
	// step OK
	NSteps++
	setLastErr(err)
	setMaxTorque(dy1)
	//} else {
	// undo bad step
	//	util.Assert(FixDt == 0)
	//	Time = t0
	//	data.Copy(y, y0)
	//	NUndone++
	//}
}

func (s *BackwardEuler) Free() {
	s.dy1.Free()
	s.dy1 = nil
}
