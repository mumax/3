package cuda

// General solver utils.

import (
	"code.google.com/p/mx3/util"
	"log"
)

// embedded in all solvers
type solverCommon struct {
	Dt_si, dt_mul          float64  // time step = dt_si (seconds) *dt_mul, which should be nice float32
	time                   *float64 // in seconds
	Mindt, Maxdt           float64  // minimum and maximum time step
	MaxErr, Headroom       float64  // maximum error per step
	LastErr                float64  // error of last step
	NSteps, NUndone, NEval int      // number of good steps, undone steps
	FixDt                  bool     // fixed time step?
}

func newSolverCommon(dt_si, dt_mul float64, time *float64) solverCommon {
	return solverCommon{time: time, Dt_si: dt_si, dt_mul: dt_mul,
		MaxErr: 1e-4, Headroom: 0.75} // TODO: use consts
}

// adapt time step: dt *= corr, but limited to sensible values.
func (e *solverCommon) adaptDt(corr float64) {
	if e.FixDt {
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
	e.Dt_si *= corr
	if e.Mindt != 0 && e.Dt_si < e.Mindt {
		e.Dt_si = e.Mindt
	}
	if e.Maxdt != 0 && e.Dt_si > e.Maxdt {
		e.Dt_si = e.Maxdt
	}
	if e.Dt_si == 0 {
		log.Fatal("time step too small")
	}
}
