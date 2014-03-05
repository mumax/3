package engine

import (
	"github.com/mumax/3/data"
	"github.com/mumax/3/util"
)

var (
	solverPostStep         func()            = M.normalize // called on y after successful step, typically normalizes magnetization
	Dt_si                  float64           = 1e-15       // time step = dt_si (seconds) *dt_mul, which should be nice float32
	dt_mul                 *float64          = &GammaLL    // TODO: simplify
	MinDt, MaxDt           float64                         // minimum and maximum time step
	MaxErr                 float64           = 1e-4
	Headroom               float64           = 0.75      // maximum error per step
	LastErr                float64                       // error of last step
	NSteps, NUndone, NEval int                           // number of good steps, undone steps
	FixDt                  float64                       // fixed time step?
	torqueFn               func(*data.Slice) = SetTorque // writes torque to dst // TODO: rm
	stepper                func(*data.Slice)             // generic step, can be EulerStep, HeunStep, etc
)

//func NewSolver(torqueFn func(dst *data.Slice), postStep func(), dt_si float64, dt_mul *float64, step func(*solver, *data.Slice)) solver {
//	util.Argument(dt_si > 0 && *dt_mul > 0)
//	return solver{torqueFn: torqueFn, postStep: postStep,
//		Dt_si: dt_si, dt_mul: dt_mul,
//		MaxErr: 1e-4, Headroom: 0.75, step: step}
//}

// TODO: rm
func Step(y *data.Slice) {
	stepper(y)
}

// adapt time step: dt *= corr, but limited to sensible values.
func adaptDt(corr float64) {
	if FixDt != 0 {
		Dt_si = FixDt
		return
	}
	util.AssertMsg(corr != 0, "Time step too small, check if parameters are sensible")
	corr *= Headroom
	if corr > 2 {
		corr = 2
	}
	if corr < 1./2. {
		corr = 1. / 2.
	}
	Dt_si *= corr
	if MinDt != 0 && Dt_si < MinDt {
		Dt_si = MinDt
	}
	if MaxDt != 0 && Dt_si > MaxDt {
		Dt_si = MaxDt
	}
	if Dt_si == 0 {
		util.Fatal("time step too small")
	}
}
