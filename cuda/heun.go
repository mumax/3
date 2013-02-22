package cuda

import (
	"code.google.com/p/mx3/data"
	"code.google.com/p/mx3/util"
	"log"
)

// Adaptive heun solver.
// TODO: now only for magnetization (because it normalizes), post-step hook?
type Heun struct {
	solverCommon
	dy0      *data.Slice // internal buffer
	y, dy    *data.Slice // external
	updateDy func()      // updates dy
	init     bool
}

func NewHeun(y, dy *data.Slice, dt, multiplier float64) *Heun {
	util.Argument(dt > 0 && multiplier > 0)
	dy0 := NewSlice(3, dy.Mesh())
	return &Heun{dy0: dy0, y: y, dy: dy, solverCommon: newSolverCommon(dt, multiplier)}
}

// Run for a duration in seconds
func (e *Heun) Advance(seconds float64) {
	log.Println("heun solver:", seconds, "s")
	stop := e.time + seconds
	for e.time < stop {
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
//func (e *Heun) Relax(maxerr float64) {
//	nimble.RunStack()
//	core.Log("relax down to", maxerr, "of equilibrium")
//	LockCudaThread()
//	if maxerr < 1e-7 {
//		core.Fatalf("relax: max error too small")
//	}
//	preverr := e.Maxerr
//	e.Maxerr = 1e-2
//	for {
//		e.Step()
//		if e.delta < e.Maxerr/e.Headroom {
//			e.Maxerr /= 2
//			e.dt_si /= 1.41
//		}
//		if e.Maxerr < maxerr {
//			break
//		}
//	}
//	e.Maxerr = preverr
//}

// Take one time step
func (e *Heun) Step() {

	//dy0 := e.dy0
	dt := float32(e.dt_si * e.dt_mul) // could check here if it is in float32 ranges
	util.Assert(dt > 0)

	// stage 1
	//nimble.Clock.Send(e.time, true)
	e.updateDy()
	//	Madd2(y, y, dy, 1, dt)
	//	data.Copy(dy0, dy)
	//
	//	// stage 2
	//	//nimble.Clock.Send(e.time+e.dt_si, false)
	//	e.updateDy()
	//	{
	//		e.err = MaxVecDiff(dy0.Safe(0), dy0.Safe(1), dy0.Safe(2), dy.Safe(0), dy.Safe(1), dy.Safe(2)) * float64(dt)
	//		e.checkErr()
	//
	//		if e.err < e.Maxerr || e.dt_si <= e.Mindt { // mindt check to avoid infinite loop
	//			e.delta = MaxVecNorm(dy.Safe(0), dy.Safe(1), dy.Safe(2)) * float64(dt)
	//			Madd3(y, y, dy, dy0, 1, 0.5*dt, -0.5*dt)
	//			Normalize(y)
	//			e.time += e.dt_si
	//			e.steps++
	//			e.adaptDt(math.Pow(e.Maxerr/e.err, 1./2.))
	//		} else { // undo.
	//			e.delta = 0
	//			Madd2(y, y, dy0, 1, -dt)
	//			e.undone++
	//			e.adaptDt(math.Pow(e.Maxerr/e.err, 1./3.))
	//		}
	//		e.sendDebugOutput()
	//		e.updateDash()
	//	}
}
