package engine

import (
	"code.google.com/p/mx3/util"
	"log"
)

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

var pause = false

func Pause()       { pause = true }
func Paused() bool { return pause }
func init() {
	parser.AddFunc("pause", Pause)
}

// Runs as long as condition returns true.
func RunCond(condition func() bool) {
	checkInited() // todo: check in handler
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
