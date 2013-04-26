package engine

// Handlers for web control "ctl/"

import (
	"fmt"
	"net/http"
	"os"
	"strconv"
	"time"
)

var pause = false

// inject to pause simulation.
func pauseFn() { pause = true }

func control(w http.ResponseWriter, r *http.Request) {
	cmd := r.URL.Path[len("/ctl/"):]
	arg := r.FormValue("value")

	switch cmd {
	default:
		http.Error(w, "unhandled control: "+cmd, http.StatusNotFound)
		return

	case "pause":
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
		inject <- pauseFn
		inject <- func() { Steps(v) }

	case "kill":
		os.Exit(0)
	}

	http.Redirect(w, r, "/", http.StatusFound)
}

// when we last saw browser activity
var lastKeepalive time.Time

func isrunning(w http.ResponseWriter, r *http.Request) {
	lastKeepalive = time.Now()
	fmt.Fprint(w, !pause)
}
