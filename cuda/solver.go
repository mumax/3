package cuda

// General solver utils.

import (
	"code.google.com/p/mx3/util"
	"log"
)

// embedded in all solvers
type solverCommon struct {
	dt_si, dt_mul    float64  // time step = dt_si (seconds) *dt_mul, which should be nice float32
	Time             *float64 // in seconds
	Mindt, Maxdt     float64  // minimum and maximum time step
	Maxerr, Headroom float64  // maximum error per step
	NSteps, undone   int      // number of good steps, undone steps
	Fixdt            bool     // fixed time step?
}

func newSolverCommon(dt_si, dt_mul float64, time *float64) solverCommon {
	return solverCommon{dt_si: dt_si, dt_mul: dt_mul,
		Maxerr: 1e-4, Headroom: 0.75}
}

// adapt time step: dt *= corr, but limited to sensible values.
func (e *solverCommon) adaptDt(corr float64) {
	if e.Fixdt {
		return
	}
	util.Assert(corr != 0)
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

func solverCheckErr(err float64) {
	// Note: err == 0 occurs when input is NaN (or time step massively too small).
	if err == 0 {
		util.DashExit()
		log.Fatalf("solver: cannot adapt dt")
	}
}
