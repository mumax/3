package gpu

import (
	"code.google.com/p/nimble-cube/core"
	"code.google.com/p/nimble-cube/nimble"
	"github.com/barnex/cuda5/cu"
)

// Euler solver.
type Euler struct {
	y      nimble.ChanN
	dy     nimble.RChanN
	dt     float32
	init   bool
	stream cu.Stream
}

func NewEuler(y nimble.ChanN, dy_ nimble.ChanN, dt, multiplier float64) *Euler {
	dy := dy_.NewReader()
	return &Euler{y, dy, float32(dt * multiplier), false, cu.StreamCreate()}
}

func (e *Euler) Steps(steps int) {
	core.Log("GPU euler solver:", steps, "steps")
	n := e.y.Mesh().NCell()

	// Send out initial value
	if !e.init {
		e.y.WriteNext(n)
		e.init = true
	}

	e.y.WriteDone()

	for s := 0; s < steps-1; s++ {
		dy := Device3(e.dy.ReadNext(n))
		y := Device3(e.y.WriteNext(n))
		rotatevec(y, dy, e.dt, e.stream)
		e.y.WriteDone()
		e.dy.ReadDone()
	}
	// gentle hack:
	// do not split the last frame in blocks ans do not signal writeDone
	// then the pipeline comes in a consistent state before Steps() returns.
	dy := Device3(e.dy.ReadNext(n))
	y := Device3(e.y.WriteNext(n))
	rotatevec(y, dy, e.dt, e.stream)
	e.dy.ReadDone()
}
