package cuda

// General solver utils.

import (
	"github.com/mumax/3/data"
	"github.com/mumax/3/util"
	"log"
)

type Solver struct {
	y                      *data.Slice       // the quantity to be time stepped
	torqueFn               func(*data.Slice) // updates dy
	postStep               func(*data.Slice) // called on y after successful step, typically normalizes magnetization
	Dt_si, dt_mul          float64           // time step = dt_si (seconds) *dt_mul, which should be nice float32
	time                   *float64          // in seconds
	MinDt, MaxDt           float64           // minimum and maximum time step
	MaxErr, Headroom       float64           // maximum error per step
	LastErr                float64           // error of last step
	NSteps, NUndone, NEval int               // number of good steps, undone steps
	FixDt                  float64           // fixed time step?
	step                   func(*Solver)     //
}

func NewSolver(y *data.Slice, torqueFn, postStep func(*data.Slice), dt_si, dt_mul float64, time *float64, step func(*Solver)) Solver {
	util.Argument(dt_si > 0 && dt_mul > 0)
	return Solver{y: y, torqueFn: torqueFn, postStep: postStep,
		time: time, Dt_si: dt_si, dt_mul: dt_mul,
		MaxErr: 1e-4, Headroom: 0.75, step: step}
}

func (e *Solver) Step() {
	e.step(e)
}

// adapt time step: dt *= corr, but limited to sensible values.
func (e *Solver) adaptDt(corr float64) {
	if e.FixDt != 0 {
		e.Dt_si = e.FixDt
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
