package gpu

// General solver utils.

import (
	"code.google.com/p/mx3/core"
	"code.google.com/p/mx3/dump"
	"code.google.com/p/mx3/nimble"
	"github.com/barnex/cuda5/cu"
)

// embedded in all solvers
type solverCommon struct {
	dt_si, dt_mul    float64          // time step = dt_si (seconds) *dt_mul, which should be nice float32
	time             float64          // in seconds
	Mindt, Maxdt     float64          // minimum and maximum time step
	Maxerr, Headroom float64          // maximum error per step
	steps, undone    int              // number of good steps, undone steps
	delta, err       float64          // max delta, max error of last step
	debug            dump.TableWriter // save t, dt, error here
}

func newSolverCommon(dt_si, dt_mul float64) solverCommon {
	var w dump.TableWriter
	if core.DEBUG {
		w = dump.NewTableWriter(core.OpenFile(core.OD+"/debug_heun.table"),
			[]string{"t", "dt", "err", "delta"}, []string{"s", "s", "", ""})
	}

	return solverCommon{dt_si: dt_si, dt_mul: dt_mul,
		Maxerr: 1e-4, Headroom: 0.75,
		debug: w}
}

// y += dy * dt
//func maddvec(y, dy [3]safe.Float32s, dt float32) {
//	for i := 0; i < 3; i++ {
//		Madd2(y[i], y[i], dy[i], 1, dt)
//	}
//}

// dst += a * s1 + b * s2
//func madd2vec(dst, a, b [3]safe.Float32s, s1, s2 float32, str [3]cu.Stream) {
//	for i := range dst {
//		Madd3(dst[i], dst[i], a[i], b[i], 1, s1, s2)
//	}
//}

// dst = src
//func cpyvec(dst, src [3]safe.Float32s, str [3]cu.Stream) {
//	for i := 0; i < 3; i++ {
//		dst[i].CopyDtoDAsync(src[i], str[i])
//	}
//	syncAll(str[:])
//}

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
func (e *solverCommon) updateDash() {
	nimble.Dash(e.steps, e.undone, e.time, e.dt_si, e.err)
}

func (e *solverCommon) sendDebugOutput() {
	if core.DEBUG {
		e.debug.Data[0], e.debug.Data[1], e.debug.Data[2], e.debug.Data[3] = float32(e.time), float32(e.dt_si), float32(e.err), float32(e.delta)
		e.debug.WriteData()
	}
}

func (e *solverCommon) checkErr() {
	// Note: err == 0 occurs when input is NaN (or time step massively too small).
	if e.err == 0 {
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
