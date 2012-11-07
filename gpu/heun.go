package gpu

//import (
//	"github.com/barnex/cuda5/cu"
//	"github.com/barnex/cuda5/safe"
//	"nimble-cube/core"
//)
//
//// Heun solver.
//type Heun struct {
//	dy0    [3]safe.Float32s
//	y      core.Chan3
//	dy     core.RChan3
//	dt     float32
//	init   bool
//	stream cu.Stream
//}
//
//func NewHeun(y core.Chan3, dy_ core.Chan3, dt, multiplier float64) *Heun {
//	dy := dy_.NewReader()
//	dy0 := MakeVectors(core.Prod(y.Size()))
//	return &Heun{dy0, y, dy, float32(dt * multiplier), false, cu.StreamCreate()}
//}
//
//func (e *Heun) Steps(steps int) {
//	core.RunStack()
//	core.Log("GPU heun solver:", steps, "steps")
//	LockCudaThread()
//	defer UnlockCudaThread()
//
//	n := core.Prod(e.y.Size())
//
//	// Send out initial value
//	if !e.init {
//		e.y.WriteNext(n)
//		e.init = true
//	}
//
//	e.y.WriteDone()
//
//	for s := 0; s < steps; s++ {
//
//		dy := e.dy.ReadNext(n)
//		y := e.y.WriteNext(n)
//
//		rotatevec(y, dy, e.dt, e.stream)
//
//		for i := 0; i < 3; i++ {
//			e.dy0[i].CopyDtoDAsync(dy[i], e.stream)
//		}
//		e.stream.Synchronize()
//
//		e.y.WriteDone()
//		e.dy.ReadDone()
//
//		dy = e.dy.ReadNext(n)
//		y = e.y.WriteNext(n)
//
//		rotatevec2(y, dy, 0.5*e.dt, e.dy0, -0.5*e.dt, e.stream)
//
//		e.dy.ReadDone()
//		if s != steps-1 {
//			e.y.WriteDone() // do not signal write done to get pipeline in fixed state
//		}
//	}
//}
