package engine

// Handlers for web control "ctl/"

import (
	"code.google.com/p/mx3/util"
	"fmt"
	"log"
	"net/http"
	"os"
	"strconv"
)

var inject = make(chan func()) // inject function calls into the cuda main loop. Executed in between time steps.
// TODO: nil if not serving web.

var pause = false // TODO: should only start paused

func pauseFn() { pause = true }

func control(w http.ResponseWriter, r *http.Request) {
	cmd := r.URL.Path[len("/ctl/"):]
	arg := r.FormValue("value")
	ui.Msg = "" // clear last message

	switch cmd {
	default:
		http.Error(w, "unhandled control: "+cmd, http.StatusNotFound)
		return

	case "pause": // reacts immediately
		inject <- pauseFn

	case "run":
		v, err := strconv.ParseFloat(arg, 64)
		if err != nil {
			http.Error(w, cmd+":"+err.Error(), 400)
			return
		}
		ui.Runtime = v
		inject <- pauseFn
		inject <- func() { Run(v) }

	case "steps":
		v, err := strconv.Atoi(arg)
		if err != nil {
			http.Error(w, cmd+":"+err.Error(), 400)
			return
		}
		ui.Steps = v
		inject <- func() { Steps(v) }

	case "kill":
		os.Exit(0)
	}

	http.Redirect(w, r, "/", http.StatusFound)
}

func injectAndWait(task func()) {
	ready := make(chan int)
	inject <- func() { task(); ready <- 1 }
	<-ready
}

func isrunning(w http.ResponseWriter, r *http.Request) {
	fmt.Fprint(w, !pause)
}

// Enter interactive mode. Only the main thread (which does the simulation, is cuda locked)
// should enter here.
func Interactive() {
	pause = true
	log.Println("entering interactive mode")
	if webPort == "" {
		GoServe(*Flag_port)
	}

	for {
		log.Println("awaiting interaction")
		(<-inject)()
	}
}

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

// Runs as long as condition returns true.
func RunCond(condition func() bool) {
	checkInited() // todo: check in handler
	defer util.DashExit()

	pause = false
	for condition() && !pause {
		select {
		default:
			step()
		case r := <-inject:
			r()
		}
	}
	pause = true
}
