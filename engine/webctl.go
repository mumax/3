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
		ui.Lock()
		ui.pleaseStop = true
		for ui.Running {
			ui.Wait()
		}
		ui.Unlock()

	case "run":
		v, err := strconv.ParseFloat(arg, 64)
		if err != nil {
			http.Error(w, cmd+":"+err.Error(), 400)
			return
		}
		ui.Runtime = v
		requests <- req{cmd: cmd, argn: v}
		ui.Msg = <-response

	case "exit":
		os.Exit(0)
	}
	http.Redirect(w, r, "/", http.StatusFound)
}

// Enter interactive mode.
func Interactive() {
	log.Println("entering interactive mode")
	if webPort == "" {
		GoServe(*Flag_port)
	}
	for {
		r := <-requests
		log.Println("request:", r)
		switch r.cmd {
		default:
			msg := "interactive: unhandled command: " + r.cmd
			log.Println(msg)
			response <- msg
		case "run":
			msg := fmt.Sprintln("interactive run for", r.argn, "s")
			response <- msg
			log.Println(msg)
			Run(r.argn)
		}
	}
}

// Run the simulation for a number of seconds.
func Run(seconds float64) {

	log.Println("run for", seconds, "s")
	checkInited() // todo: check in handler

	stop := Time + seconds
	defer util.DashExit()

	ui.Lock()
	ui.Running = true
	ui.Unlock()

	for {
		step()
		ui.Lock()
		if Time >= stop || ui.pleaseStop {
			break
		} else {
			ui.Unlock()
		}
	}
	ui.Running = false
	ui.pleaseStop = false
	ui.Unlock()
	ui.Signal()
}

// Run the simulation for a number of steps.
func Steps(n int) {
	//	log.Println("run for", n, "steps")
	//	checkInited()
	//	defer util.DashExit()
	//
	//	ok := true
	//	for i := 0; i < n && ok; i++ {
	//		step()
	//		select {
	//		default: // keep going
	//		case <-breakrun:
	//			ok = false
	//			response <- "stopped stepping"
	//		}
	//	}
}
