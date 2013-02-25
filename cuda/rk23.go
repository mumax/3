package cuda

//import (
//	"code.google.com/p/mx3/core"
//	"code.google.com/p/mx3/nimble"
//	"github.com/barnex/cuda5/safe"
//	//"math"
//)
//
//type RK23 struct {
//	solverCommon
//	y    nimble.ChanN
//	dy   nimble.RChanN
//	y0   [3]safe.Float32s    // backup
//	k    [4][3]safe.Float32s // derivatives
//	init bool
//}
//
//func NewRK23(y nimble.ChanN, dy_ nimble.ChanN, dt, multiplier float64) *RK23 {
//	core.Assert(dt > 0 && multiplier > 0)
//	s := new(RK23)
//	s.solverCommon = newSolverCommon(dt, multiplier)
//	s.y = y
//	s.dy = dy_.NewReader()
//	s.y0 = TryMakeVectors(y.BufLen())
//	for i := range s.k {
//		s.k[i] = TryMakeVectors(y.BufLen())
//	}
//	return s
//}
//
//// Run for a duration in seconds
//func (e *RK23) Advance(seconds float64) {
//	nimble.RunStack()
//	core.Log("GPU heun solver:", seconds, "s")
//	LockCudaThread()
//	stop := e.time + seconds
//	for e.time < stop {
//		e.Step()
//	}
//	nimble.DashExit()
//
//	if core.DEBUG {
//		e.debug.Flush()
//	}
//}
//
//// Run for a number of steps
//func (e *RK23) Steps(steps int) {
//	nimble.RunStack()
//	core.Log("GPU heun solver:", steps, "steps")
//	LockCudaThread()
//	for s := 0; s < steps; s++ {
//		e.Step()
//	}
//	nimble.DashExit()
//
//	if core.DEBUG {
//		e.debug.Flush()
//	}
//}
//
//// Take one time step
//func (e *RK23) Step() {
//	panic("todo")
//	//	n := e.y.Mesh().NCell()
//	//	str := e.stream
//	//
//	//	// Send out initial value
//	//	if !e.init {
//	//		// normalize initial magnetization
//	//		M := Device3(e.y.UnsafeData())
//	//		NormalizeSync(M, str[0])
//	//		e.y.WriteNext(n)
//	//		e.init = true
//	//	}
//	//	e.y.WriteDone()
//	//
//	//	y0 := e.y0
//	//	dt := float32(e.dt_si * e.dt_mul) // could check here if it is in float32 ranges
//	//	core.Assert(dt > 0)
//	//
//	//	// stage 1
//	//	nimble.Clock.Send(e.time, true)
//	//	dy := Device3(e.dy.ReadNext(n))
//	//	y := Device3(e.y.WriteNext(n))
//	//	cpyvec(y0, y, str)
//	//	maddvec(y, dy, 0.5*dt, str)
//	//	e.time += 0.5*e.dt_si
//	//	e.y.WriteDone()
//	//	e.dy.ReadDone()
//
//	//	// stage2
//	//	dy := Device3(e.dy.ReadNext(n))
//	//	y := Device3(e.y.WriteNext(n))
//	//	maddvec(y, dy, 0.5*dt, str)
//	//	//e.y.WriteDone() // todo uncomment
//	//	e.dy.ReadDone()
//
//	//	//e.dy.ReadDone()
//	//	// no writeDone() here.
//}
