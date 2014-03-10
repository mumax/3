package engine

import (
	"github.com/mumax/3/data"
	"github.com/mumax/3/util"
)

// Solver globals
var (
	Time                    float64             // time in seconds
	pause                   = true              // set pause at any time to stop running after the current step
	postStep                []func()            // called on after every full time step
	Inject                  = make(chan func()) // injects code in between time steps. Used by web interface.
	solvertype              int
	solverPostStep          func()   = M.normalize // called on y after successful step, typically normalizes magnetization
	Dt_si                   float64  = 1e-15       // time step = dt_si (seconds) *dt_mul, which should be nice float32
	dt_mul                  *float64 = &GammaLL    // TODO: simplify
	MinDt, MaxDt            float64                // minimum and maximum time step
	MaxErr                  float64  = 1e-5        // maximum error/step
	Headroom                float64  = 0.75        // maximum error per step
	LastErr                 float64                // error of last step
	NSteps, NUndone, NEvals int                    // number of good steps, undone steps
	FixDt                   float64                // fixed time step?
	stepper                 Stepper                // generic step, can be EulerStep, HeunStep, etc
)

func init() {
	DeclFunc("Run", Run, "Run the simulation for a time in seconds")
	DeclFunc("Steps", Steps, "Run the simulation for a number of time steps")
	DeclFunc("RunWhile", RunWhile, "Run while condition function is true")
	DeclFunc("SetSolver", SetSolver, "Set solver type. 1:Euler, 2:Heun")
	DeclVar("t", &Time, "Total simulated time (s)")
	DeclVar("step", &NSteps, "Total number of time steps taken")
	DeclROnly("dt", &Dt_si, "Last solver time step (s)")
	DeclFunc("NEval", getNEval, "Total number of torque evaluations")
	DeclVar("MinDt", &MinDt, "Minimum time step the solver can take (s)")
	DeclVar("MaxDt", &MaxDt, "Maximum time step the solver can take (s)")
	DeclVar("MaxErr", &MaxErr, "Maximum error per step the solver can tolerate")
	DeclVar("Headroom", &Headroom, "Solver headroom")
	DeclVar("FixDt", &FixDt, "Set a fixed time step, 0 disables fixed step")
	SetSolver(BOGAKISHAMPINE)
}

// Time stepper like Euler, Heun, RK23
type Stepper interface {
	Step() // take time step using solver globals
	Free() // free resources, if any (e.g.: RK23 previous torque)
}

// Arguments for SetSolver
const (
	EULER          = 1
	HEUN           = 2
	BOGAKISHAMPINE = 3
)

func SetSolver(typ int) {
	// free previous solver, if any
	if stepper != nil {
		stepper.Free()
	}
	switch typ {
	default:
		util.Fatalf("SetSolver: unknown solver type: %v", typ)
	case EULER:
		stepper = new(Euler)
	case HEUN:
		stepper = new(Heun)
	case BOGAKISHAMPINE:
		stepper = new(RK23)
	}
	solvertype = typ
}

// write torque to dst and increment NEvals
func torqueFn(dst *data.Slice) {
	SetTorque(dst)
	NEvals++
}

// returns number of torque evaluations
func getNEval() int {
	return NEvals
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

// Run the simulation for a number of seconds.
func Run(seconds float64) {
	stop := Time + seconds
	RunWhile(func() bool { return Time < stop })
}

// Run the simulation for a number of steps.
func Steps(n int) {
	stop := NSteps + n
	RunWhile(func() bool { return NSteps < stop })
}

// Runs as long as condition returns true.
func RunWhile(condition func() bool) {
	pause = false
	for condition() && !pause {
		select {
		default:
			step()
		// accept tasks form Inject channel
		case f := <-Inject:
			f()
		}
	}
	pause = true
}

// Runs as long as browser is connected to gui.
func RunInteractive() {
	gui_.RunInteractive()
}

// take one time step
func step() {
	stepper.Step()
	for _, f := range postStep {
		f()
	}
	DoOutput()
}

// Register function f to be called after every time step.
// Typically used, e.g., to manipulate the magnetization.
func PostStep(f func()) {
	postStep = append(postStep, f)
}

// inject code into engine and wait for it to complete.
func InjectAndWait(task func()) {
	ready := make(chan int)
	Inject <- func() { task(); ready <- 1 }
	<-ready
}
