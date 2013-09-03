package cuda

import (
	"code.google.com/p/mx3/data"
	"code.google.com/p/mx3/util"
	"math"
)

// Adaptive heun solver for vectors.
type Heun struct {
	solverCommon
	y        *data.Slice        // the quantity to be time stepped
	torqueFn func() *data.Slice // updates dy
	postStep func(*data.Slice)  // called on y after successful step, typically normalizes magnetization
}

func NewHeun(y *data.Slice, torqueFn func() *data.Slice, postStep func(*data.Slice), dt, multiplier float64, time *float64) *Heun {
	util.Argument(dt > 0 && multiplier > 0)
	return &Heun{newSolverCommon(dt, multiplier, time), y, torqueFn, postStep}
}

// Take one time step
func (e *Heun) Step() {
	dy0 := GetBuffer(3, e.y.Mesh())
	defer RecycleBuffer(dy0)

	dt := float32(e.Dt_si * e.dt_mul) // could check here if it is in float32 ranges
	util.Assert(dt > 0)

	// stage 1
	{
		dy := e.torqueFn()
		e.NEval++
		y := e.y
		Madd2(y, y, dy, 1, dt) // y = y + dt * dy
		data.Copy(dy0, dy)
	}

	// stage 2
	{
		*e.time += e.Dt_si
		dy := e.torqueFn()
		e.NEval++

		err := 0.0
		if e.FixDt == 0 { // time step not fixed
			err = MaxVecDiff(dy0, dy) * float64(dt)
		}

		y := e.y
		if err < e.MaxErr || e.Dt_si <= e.MinDt { // mindt check to avoid infinite loop
			// step OK
			Madd3(y, y, dy, dy0, 1, 0.5*dt, -0.5*dt)
			e.postStep(y)
			e.NSteps++
			e.adaptDt(math.Pow(e.MaxErr/err, 1./2.))
			e.LastErr = err
		} else {
			// undo bad step
			util.Assert(e.FixDt == 0)
			*e.time -= e.Dt_si
			Madd2(y, y, dy0, 1, -dt)
			e.NUndone++
			e.adaptDt(math.Pow(e.MaxErr/err, 1./3.))
		}
	}
}
