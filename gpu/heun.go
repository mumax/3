package gpu

import (
	"code.google.com/p/nimble-cube/core"
	"code.google.com/p/nimble-cube/nimble"
	"github.com/barnex/cuda5/cu"
	"github.com/barnex/cuda5/safe"
	"math"
)

// Heun solver.
type Heun struct {
	dy0           [3]safe.Float32s
	y             nimble.ChanN
	dy            nimble.RChanN
	dt_si, dt_mul float64 // time step = dt_si*dt_mul, which should be nice float32
	init          bool
	stream        cu.Stream
	mindt, maxdt float64
	maxerr float64
}

func NewHeun(y nimble.ChanN, dy_ nimble.ChanN, dt, multiplier float64) *Heun {
	dy := dy_.NewReader()
	dy0 := MakeVectors(y.BufLen()) // TODO: proper len?
	return &Heun{dy0, y, dy, dt, multiplier, false, cu.StreamCreate(), 0, 0, 1e-3}
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

	dy0 := e.dy0

	for s := 0; s < steps; s++ {
		dt := float32(e.dt_si * e.dt_mul) // could check here if it is in float32 ranges

		// stage 1
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
		dy = Device3(e.dy.ReadNext(n))
		y = Device3(e.y.WriteNext(n))
		{
			err := reduceMaxVecDiff(dy0[0], dy0[1], dy0[2], dy[0], dy[1], dy[2], e.stream)
			e.stream.Synchronize()
			core.Log("error:", err)

			corr := 1.
			if err < e.maxerr{
				rotatevec2(y, dy, 0.5*dt, dy0, -0.5*dt, e.stream)
				//t += dt
				corr = math.Pow(e.maxerr / err, 1./2.)
			}else{
				corr = math.Pow(e.maxerr / err, 1./3.)
			}
			e.adaptDt(corr)
			core.Log("dt:", e.dt_si)
		}
		e.dy.ReadDone()
		if s != steps-1 {
			e.y.WriteDone() // do not signal write done to get pipeline in fixed state
		}
	}
}

func(e*Heun)adaptDt(corr float64){
	if corr > 2{corr = 2}
	if corr < 1./2.{corr = 1./2.}
	e.dt_si *= corr
	if e.mindt != 0 && e.dt_si < e.mindt{
		e.dt_si = e.mindt
	}
	if e.maxdt != 0 && e.dt_si > e.maxdt{
		e.dt_si = e.maxdt
	}
}
