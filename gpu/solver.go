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

// dst += a * s1 + b * s2
func madd2vec(dst, a, b [3]safe.Float32s, s1, s2 float32, str [3]cu.Stream) {
	Madd3Async(dst[0], dst[0], a[0], b[0], 1, s1, s2, str[0])
	Madd3Async(dst[1], dst[1], a[1], b[1], 1, s1, s2, str[1])
	Madd3Async(dst[2], dst[2], a[2], b[2], 1, s1, s2, str[2])
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

// report time, dt, error to terminal dashboard
func (e *solverCommon) updateDash(err float64) {
	nimble.Dash(e.steps, e.undone, e.time, e.dt_si, err)
}

func (e *solverCommon) sendDebugOutput(err float64) {
	if core.DEBUG {
		e.debug.Data[0], e.debug.Data[1], e.debug.Data[2] = float32(e.time), float32(e.dt_si), float32(err)
		e.debug.WriteData()
	}
}

func (e *solverCommon) checkErr(err float64) {
	// Note: err == 0 occurs when input is NaN (or time step massively too small).
	if err == 0 {
		nimble.DashExit()
		core.Fatalf("solver: cannot adapt dt")
	}
}

func stream3Create() [3]cu.Stream {
	return [3]cu.Stream{cu.StreamCreate(), cu.StreamCreate(), cu.StreamCreate()}
}

func syncAll(streams []cu.Stream) {
	for _, s := range streams {
		s.Synchronize()
	}
}

