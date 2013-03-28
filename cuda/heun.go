package cuda

import (
	"code.google.com/p/mx3/data"
	"code.google.com/p/mx3/util"
	"math"
)

// Adaptive heun solver for vectors.
type Heun struct {
	solverCommon
	y        *data.Synced            // the quantity to be time stepped
	dy0      *data.Slice             // time derivative of y <- buffer could be released after step?
	torqueFn func(bool) *data.Synced // updates dy
}

func NewHeun(y *data.Synced, torqueFn func(bool) *data.Synced, dt, multiplier float64, time *float64) *Heun {
	util.Argument(dt > 0 && multiplier > 0)
	m := y.Mesh()
	dy0 := NewSlice(3, m)
	return &Heun{newSolverCommon(dt, multiplier, time), y, dy0, torqueFn}
}

// Take one time step
func (e *Heun) Step() {
	dy0 := e.dy0
	dt := float32(e.Dt_si * e.dt_mul) // could check here if it is in float32 ranges
	util.Assert(dt > 0)

	// stage 1
	{
		Dy := e.torqueFn(true) // <- hook here for output, always good step output
		dy := Dy.Read()
		y := e.y.Write()
		Madd2(y, y, dy, 1, dt) // y = y + dt * dy
		e.y.WriteDone()
		data.Copy(dy0, dy)
		Dy.ReadDone()
	}

	// stage 2
	{
		*e.Time += e.Dt_si
		Dy := e.torqueFn(false)
		dy := Dy.Read()

		err := 0.0
		if !e.Fixdt {
			err = MaxVecDiff(dy0, dy) * float64(dt)
			solverCheckErr(err)
		}

		y := e.y.Write()
		if err < e.Maxerr || e.Dt_si <= e.Mindt { // mindt check to avoid infinite loop
			// step OK
			Madd3(y, y, dy, dy0, 1, 0.5*dt, -0.5*dt)
			e.NSteps++
			e.adaptDt(math.Pow(e.Maxerr/err, 1./2.))
			e.LastErr = err
		} else {
			// undo bad step
			util.Assert(!e.Fixdt)
			*e.Time -= e.Dt_si
			Madd2(y, y, dy0, 1, -dt)
			e.NUndone++
			e.adaptDt(math.Pow(e.Maxerr/err, 1./3.))
		}
		e.y.WriteDone()
		Dy.ReadDone()
	}
}

// Run for a duration in seconds
//func (e *Heun) Advance(seconds float64) {
//	log.Println("heun solver:", seconds, "s")
//	stop := *e.Time + seconds
//	for *e.Time < stop {
//		e.Step()
//	}
//	util.DashExit()
//}

// Run for a number of steps
//func (e *Heun) Steps(steps int) {
//	log.Println("heun solver:", steps, "steps")
//	for s := 0; s < steps; s++ {
//		e.Step()
//	}
//	util.DashExit()
//}

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
