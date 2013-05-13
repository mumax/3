package cuda

import (
	"code.google.com/p/mx3/data"
	"code.google.com/p/mx3/util"
	"math"
)

// Adaptive heun solver for vectors.
type Heun struct {
	solverCommon
	y        *data.Slice            // the quantity to be time stepped
	torqueFn func(bool) *data.Slice // updates dy
	postStep func(*data.Slice)      // called on y after successful step, typically normalizes magnetization
}

func NewHeun(y *data.Slice, torqueFn func(bool) *data.Slice, postStep func(*data.Slice), dt, multiplier float64, time *float64) *Heun {
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
		dy := e.torqueFn(true) // <- hook here for output, always good step output
		e.NEval++
		y := e.y
		Madd2(y, y, dy, 1, dt) // y = y + dt * dy
		data.Copy(dy0, dy)
	}

	// stage 2
	{
		*e.time += e.Dt_si
		dy := e.torqueFn(false)
		e.NEval++

		err := 0.0
		if !e.Fixdt {
			err = MaxVecDiff(dy0, dy) * float64(dt)
			solverCheckErr(err)
		}

		y := e.y
		if err < e.MaxErr || e.Dt_si <= e.Mindt { // mindt check to avoid infinite loop
			// step OK
			Madd3(y, y, dy, dy0, 1, 0.5*dt, -0.5*dt)
			e.postStep(y)
			e.NSteps++
			e.adaptDt(math.Pow(e.MaxErr/err, 1./2.))
			e.LastErr = err
		} else {
			// undo bad step
			util.Assert(!e.Fixdt)
			*e.time -= e.Dt_si
			Madd2(y, y, dy0, 1, -dt)
			e.NUndone++
			e.adaptDt(math.Pow(e.MaxErr/err, 1./3.))
		}
	}
}

// Run until we are only maxerr away from equilibrium.
// Typ. maxerr: 1e-7 (cannot go lower).
// Run for at most maxSteps to avoid infinite loop if we fail to relax.
//func (e *Heun) Relax(maxerr float64, maxSteps int) {
//	log.Println("relax down to", maxerr, "of equilibrium")
//	if maxerr < 1e-7 {
//		log.Fatal("relax: max error too small")
//	}
//	preverr := e.Maxerr
//	e.Maxerr = 1e-2
//
//	var i int
//	for i = 0; i < maxSteps; i++ {
//		e.Step()
//		if e.delta < e.Maxerr/e.Headroom {
//			e.Maxerr /= 2
//			e.dt_si /= 1.41
//		}
//		if e.Maxerr < maxerr {
//			break
//		}
//	}
//	if i == maxSteps {
//		log.Fatalf("relax: did not converge within %v time steps.")
//	}
//	e.Maxerr = preverr
//	util.DashExit()
//}
