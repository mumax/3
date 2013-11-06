package engine

import (
	"github.com/mumax/3/data"
	"github.com/mumax/3/util"
	"log"
)

type solver struct {
	y                      *data.Slice       // the quantity to be time stepped
	torqueFn               func(*data.Slice) // updates dy
	postStep               func(*data.Slice) // called on y after successful step, typically normalizes magnetization
	Dt_si, dt_mul          float64           // time step = dt_si (seconds) *dt_mul, which should be nice float32
	MinDt, MaxDt           float64           // minimum and maximum time step
	MaxErr, Headroom       float64           // maximum error per step
	LastErr                float64           // error of last step
	NSteps, NUndone, NEval int               // number of good steps, undone steps
	FixDt                  float64           // fixed time step?
	step                   func(*solver)     // generic step, can be EulerStep, HeunStep, etc
}

func NewSolver(y *data.Slice, torqueFn, postStep func(*data.Slice), dt_si, dt_mul float64, step func(*solver)) solver {
	util.Argument(dt_si > 0 && dt_mul > 0)
	return solver{y: y, torqueFn: torqueFn, postStep: postStep,
		Dt_si: dt_si, dt_mul: dt_mul,
		MaxErr: 1e-4, Headroom: 0.75, step: step}
}

func (e *solver) Step() {
	e.step(e)
}

// adapt time step: dt *= corr, but limited to sensible values.
func (e *solver) adaptDt(corr float64) {
	if e.FixDt != 0 {
		e.Dt_si = e.FixDt
		return
	}
	util.AssertMsg(corr != 0, "Time step too small, check if parameters are sensible")
	corr *= e.Headroom
	if corr > 2 {
		corr = 2
	}
	if corr < 1./2. {
		corr = 1. / 2.
	}
	e.Dt_si *= corr
	if e.MinDt != 0 && e.Dt_si < e.MinDt {
		e.Dt_si = e.MinDt
	}
	if e.MaxDt != 0 && e.Dt_si > e.MaxDt {
		e.Dt_si = e.MaxDt
	}
	if e.Dt_si == 0 {
		log.Fatal("time step too small")
	}
}
