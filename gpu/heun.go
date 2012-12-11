package gpu

import (
	"code.google.com/p/mx3/core"
	"code.google.com/p/mx3/nimble"
	"github.com/barnex/cuda5/safe"
	"math"
)

// Adaptive heun solver.
// TODO: now only for magnetization (because it normalizes), post-step hook?
type Heun struct {
	solverCommon
	dy0  [3]safe.Float32s // buffer dy/dt
	y    nimble.ChanN
	dy   nimble.RChanN
	init bool
}

func NewHeun(y nimble.ChanN, dy_ nimble.ChanN, dt, multiplier float64) *Heun {
	core.Assert(dt > 0 && multiplier > 0)
	dy := dy_.NewReader()
	dy0 := TryMakeVectors(y.BufLen()) // TODO: proper len?
	return &Heun{dy0: dy0, y: y, dy: dy, solverCommon: newSolverCommon(dt, multiplier)}
}

// Run for a duration in seconds
func (e *Heun) Advance(seconds float64) {
	nimble.RunStack()
	core.Log("GPU heun solver:", seconds, "s")
	LockCudaThread()
	stop := e.time + seconds
	for e.time < stop {
		e.Step()
	}
	nimble.DashExit()

	if core.DEBUG {
		e.debug.Flush()
	}
}

// Run for a number of steps
func (e *Heun) Steps(steps int) {
	nimble.RunStack()
	core.Log("GPU heun solver:", steps, "steps")
	LockCudaThread()
	for s := 0; s < steps; s++ {
		e.Step()
	}
	nimble.DashExit()

	if core.DEBUG {
		e.debug.Flush()
	}
}

// Run until we are only maxerr away from equilibrium.
// Typ. maxerr: 1e-7 (cannot go lower).
func (e *Heun) Relax(maxerr float64) {
	nimble.RunStack()
	core.Log("relax down to", maxerr, "of equilibrium")
	LockCudaThread()
	if maxerr < 1e-7 {
		core.Fatalf("relax: max error too small")
	}
	preverr := e.Maxerr
	e.Maxerr = 1e-2
	for {
		e.Step()
		if e.delta < e.Maxerr/e.Headroom {
			e.Maxerr /= 2
			e.dt_si /= 1.41
		}
		if e.Maxerr < maxerr {
			break
		}
	}
	e.Maxerr = preverr
}

// Take one time step
func (e *Heun) Step() {
	n := e.y.Mesh().NCell()
	str := e.stream

	// Send out initial value
	if !e.init {
		// normalize initial magnetization
		M := Device3(e.y.UnsafeData())
		NormalizeSync(M, str[0])
		e.y.WriteNext(n)
		e.init = true
	}
	e.y.WriteDone()

	dy0 := e.dy0
	dt := float32(e.dt_si * e.dt_mul) // could check here if it is in float32 ranges
	core.Assert(dt > 0)

	// stage 1
	nimble.Clock.Send(e.time, true)
	dy := Device3(e.dy.ReadNext(n))
	y := Device3(e.y.WriteNext(n))
	maddvec(y, dy, dt, str)
	e.y.WriteDone()
	cpyvec(dy0, dy, str)
	e.dy.ReadDone()

	// stage 2
	nimble.Clock.Send(e.time+e.dt_si, false)
	dy = Device3(e.dy.ReadNext(n))
	y = Device3(e.y.WriteNext(n))
	{
		e.err = MaxVecDiff(dy0[0], dy0[1], dy0[2], dy[0], dy[1], dy[2], str[0]) * float64(dt)
		e.checkErr()

		if e.err < e.Maxerr || e.dt_si <= e.Mindt { // mindt check to avoid infinite loop
			e.delta = MaxVecNorm(dy[0], dy[1], dy[2], str[0]) * float64(dt)
			madd2vec(y, dy, dy0, 0.5*dt, -0.5*dt, str)
			NormalizeSync(y, str[0])
			e.time += e.dt_si
			e.steps++
			e.adaptDt(math.Pow(e.Maxerr/e.err, 1./2.))
		} else { // undo.
			e.delta = 0
			maddvec(y, dy0, -dt, str)
			e.undone++
			e.adaptDt(math.Pow(e.Maxerr/e.err, 1./3.))
		}
		e.sendDebugOutput()
		e.updateDash()
	}
	e.dy.ReadDone()
	// no writeDone() here.
}
