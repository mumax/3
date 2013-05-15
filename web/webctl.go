package engine

// Handlers for web control "ctl/"

import (
	"fmt"
	"log"
	"net/http"
	"time"
)

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
