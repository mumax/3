package gpu

import (
	"github.com/barnex/cuda5/cu"
	"nimble-cube/core"
)

// Euler solver.
type Euler struct {
	y      core.Chan3
	dy     core.RChan3
	dt     float32
	init   bool
	stream cu.Stream
}

func NewEuler(y core.Chan3, dy core.RChan3, dt, multiplier float64) *Euler {
	return &Euler{y, dy, float32(dt * multiplier), false, cu.StreamCreate()}
}

func (e *Euler) Steps(steps int) {
	core.Log("GPU euler solver:", steps, "steps")
	n := core.Prod(e.y.Size())

	// Send out initial value
	if !e.init {
		e.y.WriteNext(n)
		e.init = true
	}

	e.y.WriteDone()

	for s := 0; s < steps-1; s++ {
		dy := e.dy.ReadNext(n)
		y := e.y.WriteNext(n)
		rotatevec(y, dy, e.dt, e.stream)
		e.y.WriteDone()
		e.dy.ReadDone()
	}
	// gentle hack:
	// do not split the last frame in blocks ans do not signal writeDone
	// then the pipeline comes in a consistent state before Steps() returns.
	dy := e.dy.ReadNext(n)
	y := e.y.WriteNext(n)
	rotatevec(y, dy, e.dt, e.stream)
	e.dy.ReadDone()
}
