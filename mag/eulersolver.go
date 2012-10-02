package mag

import (
	"math"
	"nimble-cube/core"
)

// Euler solver.
type Euler struct {
	y        core.Chan3
	dy       core.RChan3
	dt       float32
	blocklen int
	init     bool
}

func NewEuler(y core.Chan3, dy core.RChan3, dt float32) *Euler {
	return &Euler{y, dy, dt, core.BlockLen(y.Size()), false}
}

func (e *Euler) Steps(steps int) {
	core.Log("euler solver:", steps, "steps")
	n := core.Prod(e.y.Size())
	block := e.blocklen
	dt := e.dt

	// Send out initial value
	if !e.init {
		e.y.WriteNext(n)
		e.y.WriteDone()
		e.init = true
	}

	for s := 0; s < steps; s++ {
		for I := 0; I < n; I += block {

			dy := e.dy.ReadNext(block)
			y := e.y.WriteNext(block)

			for i := range y[0] {
				var y1 Vector
				y1[X] = y[X][i] + dt*dy[X][i]
				y1[Y] = y[Y][i] + dt*dy[Y][i]
				y1[Z] = y[Z][i] + dt*dy[Z][i]

				inorm := 1 / float32(math.Sqrt(float64(y1[X]*y1[X]+y1[Y]*y1[Y]+y1[Z]*y1[Z])))

				y[X][i] = inorm * y1[X]
				y[Y][i] = inorm * y1[Y]
				y[Z][i] = inorm * y1[Z]
			}
			e.y.WriteDone() // TODO: not on last step?
			e.dy.ReadDone()
		}
	}
}
