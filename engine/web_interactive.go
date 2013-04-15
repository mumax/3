package engine

import (
	"fmt"
	"log"
	"net/http"
	"os"
	"strconv"
)

var (
	requests = make(chan req) // requests for long-running commands are sent here
	response = make(chan string)
	breakrun = make(chan bool) // breaks run loops
)

type req struct {
	cmd   string
	value string
	nval  float64
}

func control(w http.ResponseWriter, r *http.Request) {
	cmd := r.URL.Path[len("/ctl/"):]
	val := r.FormValue("value")

	guis.Msg = ""
	switch cmd {
	default:
		http.Error(w, "illegal control: "+cmd, http.StatusNotFound)
		return
	case "exit":
		os.Exit(0)
	case "break":
		breakrun <- true
		guis.Msg = <-response
	case "run":
		v, err := strconv.ParseFloat(val, 64)
		if err != nil {
			http.Error(w, cmd+":"+err.Error(), 400)
			return
		}
		guis.Runtime = v
		requests <- req{cmd: cmd, nval: v}
		guis.Msg = <-response
	case "steps":
		v, err := strconv.Atoi(val)
		if err != nil {
			http.Error(w, cmd+":"+err.Error(), 400)
			return
		}
		guis.Steps = int(v)
		requests <- req{cmd: cmd, nval: float64(v)}
		guis.Msg = <-response
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
			guis.Msg = "Paused"
		case "steps":
			msg := fmt.Sprintln("interactive run for", int(r.nval), "steps")
			log.Println(msg)
			response <- msg
			Steps(int(r.nval))
			guis.Msg = "Paused"
		}
	}
}
