package gpu

import (
	"github.com/barnex/cuda5/cu"
	"github.com/barnex/cuda5/safe"
	"code.google.com/p/nimble-cube/core"
	"code.google.com/p/nimble-cube/nimble"
)

// Heun solver.
type Heun struct {
	dy0    [3]safe.Float32s
	y      nimble.ChanN
	dy     nimble.RChanN
	dt     float32
	init   bool
	stream cu.Stream
}

func NewHeun(y nimble.ChanN, dy_ nimble.ChanN, dt, multiplier float64) *Heun {
	dy := dy_.NewReader()
	dy0 := MakeVectors(y.BufLen())// TODO: proper len?
	return &Heun{dy0, y, dy, float32(dt * multiplier), false, cu.StreamCreate()}
}

func (e *Heun) Steps(steps int) {
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

	for s := 0; s < steps; s++ {

		dy := Device3(e.dy.ReadNext(n))
		y := Device3(e.y.WriteNext(n))

		rotatevec(y, dy, e.dt, e.stream)

		for i := 0; i < 3; i++ {
			e.dy0[i].CopyDtoDAsync(dy[i], e.stream)
		}
		e.stream.Synchronize()

		e.y.WriteDone()
		e.dy.ReadDone()

		dy = Device3(e.dy.ReadNext(n))
		y = Device3(e.y.WriteNext(n))

		rotatevec2(y, dy, 0.5*e.dt, e.dy0, -0.5*e.dt, e.stream)

		e.dy.ReadDone()
		if s != steps-1 {
			e.y.WriteDone() // do not signal write done to get pipeline in fixed state
		}
	}
}
