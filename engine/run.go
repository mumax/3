package engine

import (
	"code.google.com/p/mx3/util"
	"log"
)

var pause = false // set pause at any time to stop running after the current step

// Run the simulation for a number of seconds.
func Run(seconds float64) {
	log.Println("run for", seconds, "s")
	stop := Time + seconds
	RunCond(func() bool { return Time < stop })
}

// Run the simulation for a number of steps.
func Steps(n int) {
	log.Println("run for", n, "steps")
	stop := Solver.NSteps + n
	RunCond(func() bool { return Solver.NSteps < stop })
}

// Pause the simulation, only useful for web gui.
func Pause() {
	pause = true
}

// Check if simulation is paused. Used by web gui.
func Paused() bool {
	return pause
}

func init() {
	world.Func("pause", Pause)
}

// Runs as long as condition returns true.
func RunCond(condition func() bool) {
	checkMesh() // todo: check in handler
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

// injects arbitrary code into the engine run loops. Used by web interface.
var Inject = make(chan func()) // inject function calls into the cuda main loop. Executed in between time steps.

// inject code into engine and wait for it to complete.
func InjectAndWait(task func()) {
	ready := make(chan int)
	Inject <- func() { task(); ready <- 1 }
	<-ready
}
