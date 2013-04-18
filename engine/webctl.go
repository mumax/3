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

var (
	requests = make(chan req)    // requests for long-running commands are sent here
	response = make(chan string) // responses to requests channel are sent here
)

// request for long-running command
type req struct {
	cmd  string  // command, e.g., "run"
	arg  string  // argument, if any, e.g. "1e-9" (s)
	argn float64 // argument as number, if applicable.
}

func control(w http.ResponseWriter, r *http.Request) {
	cmd := r.URL.Path[len("/ctl/"):]
	arg := r.FormValue("value")
	ui.Msg = "" // clear last message

	switch cmd {
	default:
		http.Error(w, "illegal control: "+cmd, http.StatusNotFound)
		return

	case "pause": // reacts immediately
		pause()

	case "run":
		// todo: should not allow run while running
		v, err := strconv.ParseFloat(arg, 64)
		if err != nil {
			http.Error(w, cmd+":"+err.Error(), 400)
			return
		}
		ui.Runtime = v
		requests <- req{cmd: cmd, argn: v}
		ui.Msg = <-response // TODO: msg is not used

	case "steps":
		// todo: should not allow run while running
		v, err := strconv.Atoi(arg)
		if err != nil {
			http.Error(w, cmd+":"+err.Error(), 400)
			return
		}
		ui.Steps = v
		requests <- req{cmd: cmd, argn: float64(v)}
		ui.Msg = <-response

	case "kill":
		os.Exit(0)
	}
	http.Redirect(w, r, "/", http.StatusFound)
}

func isrunning(w http.ResponseWriter, r *http.Request) {
	//log.Print("alive")
	fmt.Fprint(w, ui.running)
}

func pause() {
	ui.Lock()
	ui.pleaseStop = true
	for ui.running {
		ui.Wait()
	}
	ui.Unlock()
}

// Enter interactive mode.
func Interactive() {
	log.Println("entering interactive mode")
	if webPort == "" {
		GoServe(*Flag_port)
	}
	for {
		log.Println("awaiting web input")
		r := <-requests
		switch r.cmd {
		default:
			msg := "interactive: unhandled command: " + r.cmd // TODO: handle better
			log.Println(msg)
			response <- msg
		case "run":
			msg := fmt.Sprintln("interactive run for", r.argn, "s")
			response <- msg
			log.Println(msg)
			Run(r.argn)
		case "steps":
			msg := fmt.Sprintln("interactive run for", r.argn, "steps")
			response <- msg
			log.Println(msg)
			Steps(int(r.argn))
		}
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

	ui.Lock()
	ui.running = true
	ui.Unlock()

	for {
		step()
		ui.Lock()
		if !condition() || ui.pleaseStop {
			break
		} else {
			ui.Unlock()
		}
	}
	ui.running = false
	ui.pleaseStop = false
	ui.Unlock()
	ui.Signal()
}
