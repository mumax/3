package gpu

import (
	"code.google.com/p/nimble-cube/core"
	"code.google.com/p/nimble-cube/dump"
	"code.google.com/p/nimble-cube/nimble"
	"github.com/barnex/cuda5/cu"
	"github.com/barnex/cuda5/safe"
	"math"
)

// Adaptive heun solver.
type Heun struct {
	dy0              [3]safe.Float32s
	y                nimble.ChanN
	dy               nimble.RChanN
	dt_si, dt_mul    float64 // time step = dt_si*dt_mul, which should be nice float32
	time             float64
	Mindt, Maxdt     float64 // minimum and maximum time step
	Maxerr, Headroom float64 // maximum error per step
	stream           cu.Stream
	init             bool
	steps            int
	debug            dump.TableWriter // save t, dt, error here
}

func NewHeun(y nimble.ChanN, dy_ nimble.ChanN, dt, multiplier float64) *Heun {
	core.Assert(dt > 0 && multiplier > 0)
	dy := dy_.NewReader()
	dy0 := MakeVectors(y.BufLen()) // TODO: proper len?
	var w dump.TableWriter
	if core.DEBUG {
		w = dump.NewTableWriter(core.OpenFile(core.OD+"/debug_heun.dump"),
			[]string{"t", "dt", "err"}, []string{"s", "s", y.Unit()})
	}
	return &Heun{dy0: dy0, y: y, dy: dy,
		dt_si: dt, dt_mul: multiplier, Maxerr: 1e-3, Headroom: 0.75,
		debug: w, stream: cu.StreamCreate()}
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

// Take one time step
func (e *Heun) Step() {
	n := e.y.Mesh().NCell()
	// Send out initial value
	if !e.init {
		e.y.WriteNext(n)
		e.init = true
	}
	e.y.WriteDone()

	dy0 := e.dy0
	dt := float32(e.dt_si * e.dt_mul) // could check here if it is in float32 ranges

	// stage 1
	nimble.Clock.Send(e.time)
	dy := Device3(e.dy.ReadNext(n))
	y := Device3(e.y.WriteNext(n))
	{
		rotatevec(y, dy, dt, e.stream)

		for i := 0; i < 3; i++ {
			dy0[i].CopyDtoDAsync(dy[i], e.stream)
		}
		e.stream.Synchronize()
	}
	e.y.WriteDone()
	e.dy.ReadDone()

	// stage 2
	nimble.Time.Send(e.time + e.dt_si)
	dy = Device3(e.dy.ReadNext(n))
	y = Device3(e.y.WriteNext(n))
	{
		err := MaxVecDiff(dy0[0], dy0[1], dy0[2], dy[0], dy[1], dy[2], e.stream) * float64(dt)

		if core.DEBUG {
			e.debug.Data[0], e.debug.Data[1], e.debug.Data[2] = float32(e.time), float32(e.dt_si), float32(err)
			e.debug.WriteData()
		}
		if err < e.Maxerr || e.dt_si <= e.Mindt { // mindt check to avoid infinite loop
			rotatevec2(y, dy, 0.5*dt, dy0, -0.5*dt, e.stream)
			e.time += e.dt_si
			e.steps++
			e.adaptDt(math.Pow(e.Maxerr/err, 1./2.))
		} else {
			// do not advance solution, nor time
			// just try again next time with smaller dt
			e.adaptDt(math.Pow(e.Maxerr/err, 1./3.))
		}
		nimble.Dash(e.steps, e.time, e.dt_si, err)
	}
	e.dy.ReadDone()
	// no writeDone() here.
}

// adapt time step: dt *= corr, but limited to sensible values.
func (e *Heun) adaptDt(corr float64) {
	corr *= e.Headroom
	if corr > 2 {
		corr = 2
	}
	if corr < 1./2. {
		corr = 1. / 2.
	}
	e.dt_si *= corr
	if e.Mindt != 0 && e.dt_si < e.Mindt {
		e.dt_si = e.Mindt
	}
	if e.Maxdt != 0 && e.dt_si > e.Maxdt {
		e.dt_si = e.Maxdt
	}
}
