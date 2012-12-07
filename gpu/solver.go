package gpu

// General solver utils.

import (
	"code.google.com/p/nimble-cube/core"
	"code.google.com/p/nimble-cube/dump"
	"code.google.com/p/nimble-cube/nimble"
	"github.com/barnex/cuda5/cu"
	"github.com/barnex/cuda5/safe"
)

// embedded in all solvers
type solverCommon struct {
	dt_si, dt_mul    float64          // time step = dt_si (seconds) *dt_mul, which should be nice float32
	time             float64          // in seconds
	Mindt, Maxdt     float64          // minimum and maximum time step
	Maxerr, Headroom float64          // maximum error per step
	steps, undone    int              // number of good steps, undone steps
	debug            dump.TableWriter // save t, dt, error here
	stream           [3]cu.Stream
}

// y += dy * dt
func maddvec(y, dy [3]safe.Float32s, dt float32, str [3]cu.Stream) {
	for i := 0; i < 3; i++ {
		Madd2Async(y[i], y[i], dy[i], 1, dt, str[i])
	}
	syncAll(str[:])
}

// dst = src
func cpyvec(dst, src [3]safe.Float32s, str [3]cu.Stream) {
	for i := 0; i < 3; i++ {
		dst[i].CopyDtoDAsync(src[i], str[i])
	}
	syncAll(str[:])
}

// adapt time step: dt *= corr, but limited to sensible values.
func (e *solverCommon) adaptDt(corr float64) {
	core.Assert(corr != 0)
	corr *= e.Headroom
	if corr > 2 {
		corr = 2
	}
	if corr < 1./2. {
		corr = 1. / 2.
	}
	e.dt_si *= corr
	if e.Mindt != 0 && e.dt_si < e.Mindt {
		e.dt_si = e.Mindt
	}
	if e.Maxdt != 0 && e.dt_si > e.Maxdt {
		e.dt_si = e.Maxdt
	}
}

func (e *solverCommon) updateDash(err float64) {
	nimble.Dash(e.steps, e.undone, e.time, e.dt_si, err)
}
