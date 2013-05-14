package engine

// Handlers for web control "ctl/"

import (
	"fmt"
	"log"
	"net/http"
	"os"
	"strconv"
	"time"
)

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

// keep session open this long after browser inactivity
const webtimeout = 60 * time.Second

func keepBrowserAlive() {
	if time.Since(lastKeepalive) < webtimeout {
		log.Println("keeping session open to browser")
		go func() {
			for {
				if time.Since(lastKeepalive) > webtimeout {
					inject <- nop // wake up
				}
				time.Sleep(1 * time.Second)
			}
		}()
		RunInteractive()
	}
}

func nop() {}
