package cuda

import (
	"code.google.com/p/mx3/data"
	"code.google.com/p/mx3/util"
	"log"
	"math"
)

// Adaptive heun solver.
// TODO: now only for magnetization (because it normalizes), post-step hook?
type Heun struct {
	solverCommon
	y, dy0      *data.Slice
	torqueFn    func(m *data.Slice) *data.Slice // updates dy
	normalizeFn func(m *data.Slice)             // normalizes y
}

func NewHeun(y *data.Slice, torqueFn func(*data.Slice) *data.Slice, normalizeFn func(m *data.Slice), dt, multiplier float64) *Heun {
	util.Argument(dt > 0 && multiplier > 0)
	dy0 := NewSlice(3, y.Mesh())
	return &Heun{newSolverCommon(dt, multiplier), y, dy0, torqueFn, normalizeFn}
}

// Take one time step
func (e *Heun) Step() {
	y, dy0 := e.y, e.dy0
	dt := float32(e.dt_si * e.dt_mul) // could check here if it is in float32 ranges
	util.Assert(dt > 0)

	// stage 1
	//nimble.Clock.Send(e.time, true)
	dy := e.torqueFn(y)
	Madd2(y, y, dy, 1, dt) // y = y + dt * dy
	data.Copy(dy0, dy)

	// stage 2
	//nimble.Clock.Send(e.time+e.dt_si, false)
	dy = e.torqueFn(y)
	{
		e.err = MaxVecDiff(dy0, dy) * float64(dt)
		e.checkErr()

		if e.err < e.Maxerr || e.dt_si <= e.Mindt { // mindt check to avoid infinite loop
			e.delta = MaxVecNorm(dy) * float64(dt)
			Madd3(y, y, dy, dy0, 1, 0.5*dt, -0.5*dt)
			e.normalizeFn(y)
			e.Time += e.dt_si
			e.NSteps++
			e.adaptDt(math.Pow(e.Maxerr/e.err, 1./2.))
		} else { // undo.
			e.delta = 0
			Madd2(y, y, dy0, 1, -dt)
			e.undone++
			e.adaptDt(math.Pow(e.Maxerr/e.err, 1./3.))
		}
		//e.sendDebugOutput()
		//e.updateDash()
	}
}

// Run for a duration in seconds
func (e *Heun) Advance(seconds float64) {
	log.Println("heun solver:", seconds, "s")
	stop := e.Time + seconds
	for e.Time < stop {
		e.Step()
	}
	//nimble.DashExit()
	//if core.DEBUG {
	//	e.debug.Flush()
	//}
}

// Run for a number of steps
func (e *Heun) Steps(steps int) {
	log.Println("heun solver:", steps, "steps")
	for s := 0; s < steps; s++ {
		e.Step()
	}
	//nimble.DashExit()
	//if core.DEBUG {
	//	e.debug.Flush()
	//}
}

// Run until we are only maxerr away from equilibrium.
// Typ. maxerr: 1e-7 (cannot go lower).
// Run for at most maxSteps to avoid infinite loop if we fail to relax.
func (e *Heun) Relax(maxerr float64, maxSteps int) {
	log.Println("relax down to", maxerr, "of equilibrium")
	if maxerr < 1e-7 {
		log.Fatal("relax: max error too small")
	}
	preverr := e.Maxerr
	e.Maxerr = 1e-2

	var i int
	for i = 0; i < maxSteps; i++ {
		e.Step()
		if e.delta < e.Maxerr/e.Headroom {
			e.Maxerr /= 2
			e.dt_si /= 1.41
		}
		if e.Maxerr < maxerr {
			break
		}
	}
	if i == maxSteps {
		log.Fatalf("relax: did not converge within %v time steps.")
	}
	e.Maxerr = preverr
}
