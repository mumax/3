package main

import (
	"github.com/mumax/3/engine"
	"log"
	"os/exec"
	"time"
)

func Interactive() {
	log.Println("entering interactive mode")
	for {
		f := <-engine.Inject
		f()
	}
}

// Enter interactive mode. Simulation is now exclusively controlled
// by web GUI (default: http://localhost:35367)
// TODO: rename
func RunInteractive() {
	//engine.Pause()
	log.Println("entering interactive mode")
	for time.Since(engine.KeepAlive()) < timeout {
		f := <-engine.Inject
		f()
	}
	log.Println("interactive session idle: exiting")
}

// exit finished simulation this long after browser was closed
const timeout = 2 * time.Second

func keepBrowserAlive() {
	if time.Since(engine.KeepAlive()) < timeout {
		log.Println("keeping session open to browser")
		go func() {
			for {
				engine.Inject <- nop // wake up RunInteractive so it may exit
				time.Sleep(1 * time.Second)
			}
		}()
		RunInteractive()
	}
}

func nop() {}

// Try to open url in a browser. Instruct to do so if it fails.
func openbrowser(url string) {
	for _, cmd := range browsers {
		err := exec.Command(cmd, url).Start()
		if err == nil {
			log.Println("\n ====\n openend web interface in", cmd, "\n ====")
			return
		}
	}
	log.Println("\n ===== \n Please open ", url, " in a browser \n ====")
}

// list of browsers to try.
var browsers = []string{"x-www-browser", "google-chrome", "chromium-browser", "firefox", "ie", "iexplore"}
