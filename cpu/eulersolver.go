package cpu

import (
	"code.google.com/p/mx3/core"
	"code.google.com/p/mx3/nimble"
	"math"
)

// Euler solver.
type Euler struct {
	y          nimble.ChanN
	dy         nimble.RChanN
	dt_si, mul float64
	time       float64
	init       bool
}

func NewEuler(y nimble.ChanN, dy nimble.RChanN, dt, multiplier float64) *Euler {
	return &Euler{y: y, dy: dy, dt_si: dt, mul: multiplier}
}

func (e *Euler) Steps(steps int) {
	core.Log("euler solver:", steps, "steps")
	n := core.Prod(e.y.Mesh().Size())
	block := e.y.BufLen()

	// Send out initial value
	if !e.init {
		e.y.WriteNext(n)
		e.init = true
	}

	e.y.WriteDone()
	nimble.Clock.Send(e.time, true)
	dt := float32(e.dt_si * e.mul)

	for s := 0; s < steps-1; s++ {
		for I := 0; I < n; I += block {
			dy := Host3(e.dy.ReadNext(block))
			y := Host3(e.y.WriteNext(block))
			eulerStep(y, dy, dt)
			e.y.WriteDone()
			e.dy.ReadDone()
		}
		e.time += e.dt_si
		nimble.Clock.Send(e.time, true)
	}
	// gentle hack:
	// do not split the last frame in blocks ans do not signal writeDone
	// then the pipeline comes in a consistent state before Steps() returns.
	dy := Host3(e.dy.ReadNext(n))
	y := Host3(e.y.WriteNext(n))
	eulerStep(y, dy, dt)
	e.dy.ReadDone()
}

func eulerStep(y, dy [3][]float32, dt float32) {
	for i := range y[0] {
		var y1 [3]float32
		y1[0] = y[0][i] + dt*dy[0][i]
		y1[1] = y[1][i] + dt*dy[1][i]
		y1[2] = y[2][i] + dt*dy[2][i]

		inorm := 1 / float32(math.Sqrt(float64(y1[0]*y1[0]+y1[1]*y1[1]+y1[2]*y1[2])))

		y[0][i] = inorm * y1[0]
		y[1][i] = inorm * y1[1]
		y[2][i] = inorm * y1[2]
	}
}
