package engine

// Handlers for web control "ctl/"

import (
	"code.google.com/p/mx3/util"
	"fmt"
	"log"
	"net/http"
	"os"
	"strconv"
	"sync"
)

var (
	requests = make(chan req)      // requests for long-running commands are sent here
	response = make(chan string)   // responses to requests channel are sent here
	breakrun = make(chan struct{}) // dropping a value breaks run loops
)

type req struct {
	cmd   string
	value string
	nval  float64
}

var (
	runcond    = sync.NewCond(new(sync.Mutex))
	pleaseStop bool
	running    bool
)

func control(w http.ResponseWriter, r *http.Request) {
	cmd := r.URL.Path[len("/ctl/"):]
	val := r.FormValue("value")
	ui.Msg = "" // clear last message

	switch cmd {
	default:
		http.Error(w, "illegal control: "+cmd, http.StatusNotFound)
		return

	case "pause": // reacts immediately
		runcond.L.Lock()
		for running {
			pleaseStop = true
			runcond.Wait()
		}
		pleaseStop = false
		runcond.L.Unlock()

	case "break":

	case "run":
		v, err := strconv.ParseFloat(val, 64)
		if err != nil {
			http.Error(w, cmd+":"+err.Error(), 400)
			return
		}
		ui.Runtime = v
		requests <- req{cmd: cmd, nval: v}
		ui.Msg = <-response
	case "steps":
		v, err := strconv.Atoi(val)
		if err != nil {
			http.Error(w, cmd+":"+err.Error(), 400)
			return
		}
		ui.Steps = int(v)
		requests <- req{cmd: cmd, nval: float64(v)}
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
		switch r.cmd {
		default:
			msg := "interactive: unhandled command: " + r.cmd
			log.Println(msg)
			response <- msg
		case "run":
			msg := fmt.Sprintln("interactive run for", r.nval, "s")
			log.Println(msg)
			response <- msg
			Run(r.nval)
			ui.Msg = "Paused"
		case "steps":
			msg := fmt.Sprintln("interactive run for", int(r.nval), "steps")
			log.Println(msg)
			response <- msg
			Steps(int(r.nval))
			ui.Msg = "Paused"
		}
	}
}

// Run the simulation for a number of seconds.
func Run(seconds float64) {

	log.Println("run for", seconds, "s")
	checkInited() // todo: check in handler

	stop := Time + seconds
	defer util.DashExit()
	for {
		step()

		runcond.L.Lock()
		if pleaseStop || Time >= stop {
			break
		} else {
			runcond.L.Unlock()
		}
	}
	running = false
	runcond.Signal()
	runcond.L.Unlock()
}

//func isRunning() bool{
//	runlock.Lock()
//	defer runlock.Unlock()
//	return running
//}

// Run the simulation for a number of steps.
func Steps(n int) {
	log.Println("run for", n, "steps")
	checkInited()
	defer util.DashExit()

	ok := true
	for i := 0; i < n && ok; i++ {
		step()
		select {
		default: // keep going
		case <-breakrun:
			ok = false
			response <- "stopped stepping"
		}
	}
}
