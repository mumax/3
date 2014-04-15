package engine

import (
	"fmt"
	"github.com/mumax/3/data"
	"github.com/mumax/3/util"
)

// Solver globals
var (
	Time                    float64                      // time in seconds
	alarm                   float64                      // alarm clock marks end time of run, dt adaptation must not cross it!
	pause                   = true                       // set pause at any time to stop running after the current step
	postStep                []func()                     // called on after every full time step
	Inject                           = make(chan func()) // injects code in between time steps. Used by web interface.
	Dt_si                   float64  = 1e-15             // time step = dt_si (seconds) *dt_mul, which should be nice float32
	MinDt, MaxDt            float64                      // minimum and maximum time step
	MaxErr                  float64  = 1e-5              // maximum error/step
	Headroom                float64  = 0.8               // solver headroom, (Gustafsson, 1992, Control of Error and Convergence in ODE Solvers)
	lastErr, peakErr        float64                      // error of last step, highest error ever
	NSteps, NUndone, NEvals int                          // number of good steps, undone steps
	FixDt                   float64                      // fixed time step?
	stepper                 Stepper                      // generic step, can be EulerStep, HeunStep, etc
	solvertype              int
)

func init() {
	DeclFunc("Run", Run, "Run the simulation for a time in seconds")
	DeclFunc("Steps", Steps, "Run the simulation for a number of time steps")
	DeclFunc("RunWhile", RunWhile, "Run while condition function is true")
	DeclFunc("SetSolver", SetSolver, "Set solver type. 1:Euler, 2:Heun")
	DeclVar("t", &Time, "Total simulated time (s)")
	DeclVar("step", &NSteps, "Total number of time steps taken")
	DeclVar("dt", &Dt_si, "Last solver time step (s)")
	DeclROnly("lastErr", &lastErr, "Maximum error of last time step")
	DeclROnly("peakErr", &peakErr, "Maximum error over all time steps")
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
	DORMANDPRINCE  = 5
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
	case DORMANDPRINCE:
		stepper = new(RK45DP)
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

// update lastErr and peakErr
func setLastErr(err float64) {
	lastErr = err
	if err > peakErr {
		peakErr = err
	}
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

	// do not cross alarm time
	if Time < alarm && Time+Dt_si > alarm {
		Dt_si = alarm - Time
	}

	util.AssertMsg(Dt_si > 0, fmt.Sprint("Time step too small: ", Dt_si))
}

// Run the simulation for a number of seconds.
func Run(seconds float64) {
	stop := Time + seconds
	alarm = stop // don't have dt adapt to go over alarm
	RunWhile(func() bool { return Time < stop })
}

// Run the simulation for a number of steps.
func Steps(n int) {
	stop := NSteps + n
	RunWhile(func() bool { return NSteps < stop })
}

// Runs as long as condition returns true.
func RunWhile(condition func() bool) {
	SanityCheck()
	pause = false
	for condition() && !pause {
		select {
		default:
			step(true) // output = true
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
func step(output bool) {
	stepper.Step()
	for _, f := range postStep {
		f()
	}
	if output {
		DoOutput()
	}
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

func SanityCheck() {
	if Msat.isZero() {
		util.Log("Note: Msat = 0")
	}
	if Aex.isZero() {
		util.Log("Note: Aex = 0")
	}
}
