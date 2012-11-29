package gpu

import (
	"code.google.com/p/nimble-cube/core"
	"code.google.com/p/nimble-cube/nimble"
	"github.com/barnex/cuda5/cu"
	"github.com/barnex/cuda5/safe"
)

// Heun solver.
type Heun struct {
	dy0           [3]safe.Float32s
	y             nimble.ChanN
	dy            nimble.RChanN
	dt_si, dt_mul float64 // time step = dt_si*dt_mul, which should be nice float32
	init          bool
	stream        cu.Stream
}

func NewHeun(y nimble.ChanN, dy_ nimble.ChanN, dt, multiplier float64) *Heun {
	dy := dy_.NewReader()
	dy0 := MakeVectors(y.BufLen()) // TODO: proper len?
	return &Heun{dy0, y, dy, dt, multiplier, false, cu.StreamCreate()}
}

func (e *Heun) SetDt(dt float64) {
	e.dt_si = dt
}

func (e *Heun) Steps(steps int) {
	nimble.RunStack()
	core.Log("GPU heun solver:", steps, "steps")
	LockCudaThread()
	defer UnlockCudaThread()

	n := e.y.Mesh().NCell()

	// Send out initial value
	if !e.init {
		e.y.WriteNext(n)
		e.init = true
	}

	e.y.WriteDone()

	// TODO: send out time, step here

	dt := float32(e.dt_si * e.dt_mul) // could check here if it is in float32 ranges
	for s := 0; s < steps; s++ {

		dy := Device3(e.dy.ReadNext(n))
		y := Device3(e.y.WriteNext(n))

		rotatevec(y, dy, dt, e.stream)

		for i := 0; i < 3; i++ {
			e.dy0[i].CopyDtoDAsync(dy[i], e.stream)
		}
		e.stream.Synchronize()

		e.y.WriteDone()
		e.dy.ReadDone()

		dy = Device3(e.dy.ReadNext(n))
		y = Device3(e.y.WriteNext(n))

		rotatevec2(y, dy, 0.5*dt, e.dy0, -0.5*dt, e.stream)

		e.dy.ReadDone()
		if s != steps-1 {
			e.y.WriteDone() // do not signal write done to get pipeline in fixed state
		}
	}
}
