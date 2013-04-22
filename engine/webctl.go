package engine

// Handlers for web control "ctl/"

import (
	"fmt"
	"net/http"
	"os"
	"strconv"
)

// TODO: nil if not serving web.

var pause = false // TODO: should only start paused

func pauseFn() { pause = true }

func control(w http.ResponseWriter, r *http.Request) {
	cmd := r.URL.Path[len("/ctl/"):]
	arg := r.FormValue("value")

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

func isrunning(w http.ResponseWriter, r *http.Request) {
	fmt.Fprint(w, !pause)
}
