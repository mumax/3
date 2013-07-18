package engine

import (
	"code.google.com/p/mx3/cuda"
	"code.google.com/p/mx3/util"
	"log"
)

func init() {
	World.Func("Run", Run)
	World.Func("Steps", Steps)
	World.Func("Pause", Pause)
	World.Func("PostStep", PostStep)
	World.Func("RunWhile", RunWhile)
	World.Var("t", &Time)
	World.Var("Dt", &Solver.Dt_si)
	World.Var("MinDt", &Solver.Mindt)
	World.Var("MaxDt", &Solver.Maxdt)
	World.Var("MaxErr", &Solver.MaxErr)
	World.Var("Headroom", &Solver.Headroom)
	World.Var("FixDt", &Solver.FixDt)
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
	//sanityCheck()
	checkM()
	defer util.DashExit()

	GUI.SetValue("solverstatus", "running")
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
	GUI.SetValue("solverstatus", "paused")
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
