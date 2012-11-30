package cpu

//import (
//	"code.google.com/p/nimble-cube/core"
//	"code.google.com/p/nimble-cube/nimble"
//	"math"
//)
//
//// Euler solver.
//type Euler struct {
//	y    nimble.ChanN
//	dy   nimble.RChanN
//	dt   float32
//	init bool
//}
//
//func NewEuler(y nimble.ChanN, dy nimble.RChanN, dt, multiplier float64) *Euler {
//	return &Euler{y, dy, float32(dt * multiplier), false}
//}
//
//func (e *Euler) Steps(steps int) {
//	core.Log("euler solver:", steps, "steps")
//	n := core.Prod(e.y.Mesh().Size())
//	block := e.y.BufLen()
//
//	// Send out initial value
//	if !e.init {
//		e.y.WriteNext(n)
//		e.init = true
//	}
//
//	e.y.WriteDone()
//
//	for s := 0; s < steps-1; s++ {
//		for I := 0; I < n; I += block {
//			dy := Host3(e.dy.ReadNext(block))
//			y := Host3(e.y.WriteNext(block))
//			eulerStep(y, dy, e.dt)
//			e.y.WriteDone()
//			e.dy.ReadDone()
//		}
//	}
//	// gentle hack:
//	// do not split the last frame in blocks ans do not signal writeDone
//	// then the pipeline comes in a consistent state before Steps() returns.
//	dy := Host3(e.dy.ReadNext(n))
//	y := Host3(e.y.WriteNext(n))
//	eulerStep(y, dy, e.dt)
//	e.dy.ReadDone()
//}
//
//func eulerStep(y, dy [3][]float32, dt float32) {
//	for i := range y[0] {
//		var y1 Vector
//		y1[X] = y[X][i] + dt*dy[X][i]
//		y1[Y] = y[Y][i] + dt*dy[Y][i]
//		y1[Z] = y[Z][i] + dt*dy[Z][i]
//
//		inorm := 1 / float32(math.Sqrt(float64(y1[X]*y1[X]+y1[Y]*y1[Y]+y1[Z]*y1[Z])))
//
//		y[X][i] = inorm * y1[X]
//		y[Y][i] = inorm * y1[Y]
//		y[Z][i] = inorm * y1[Z]
//	}
//}
