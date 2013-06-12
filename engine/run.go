package engine

import (
	"code.google.com/p/mx3/cuda"
	"code.google.com/p/mx3/util"
	"log"
)

func init() {
	world.Func("Run", Run)
	world.Func("Steps", Steps)
	world.Func("Pause", Pause)
	world.Func("PostStep", PostStep)
	world.Func("RunWhile", RunWhile)
	world.Var("Dt", &Solver.Dt_si)
	world.Var("MinDt", &Solver.Mindt)
	world.Var("MaxDt", &Solver.Maxdt)
	world.Var("MaxErr", &Solver.MaxErr)
	world.Var("Headroom", &Solver.Headroom)
	world.Var("FixDt", &Solver.Fixdt)
}

var (
	Solver   cuda.Heun
	Time     float64             // time in seconds  // todo: hide? setting breaks autosaves
	pause    bool                // set pause at any time to stop running after the current step
	postStep []func()            // called on after every time step
	Inject   = make(chan func()) // injects code in between time steps. Used by web interface.
)

// Run the simulation for a number of seconds.
func Run(seconds float64) {
	log.Println("run for", seconds, "s")
	stop := Time + seconds
	RunWhile(func() bool { return Time < stop })
}

// Run the simulation for a number of steps.
func Steps(n int) {
	log.Println("run for", n, "steps")
	stop := Solver.NSteps + n
	RunWhile(func() bool { return Solver.NSteps < stop })
}

// Pause the simulation, only useful for web gui.
func Pause() {
	pause = true
}

// Check if simulation is paused. Used by web gui.
func Paused() bool {
	return pause
}

// Runs as long as condition returns true.
func RunWhile(condition func() bool) {
	// TODO: sanityCheck
	checkM()
	defer util.DashExit()

	pause = false
	for condition() && !pause {
		select {
		default:
			step()
		case f := <-Inject:
			f()
		}
	}
	pause = true
}

func step() {
	Solver.Step()
	for _, f := range postStep {
		f()
	}
	s := Solver
	util.Dashf("step: % 8d (%6d) t: % 12es Δt: % 12es ε:% 12e", s.NSteps, s.NUndone, Time, s.Dt_si, s.LastErr)
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
