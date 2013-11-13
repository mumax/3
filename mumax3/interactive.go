package main

import (
	"fmt"
	"github.com/mumax/3/engine"
	"os/exec"
	"time"
)

func Interactive() {
	fmt.Println("entering interactive mode")
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
	fmt.Println("entering interactive mode")
	for time.Since(engine.KeepAlive()) < timeout {
		f := <-engine.Inject
		f()
	}
	fmt.Println("browser disconnected, exiting")
}

// exit finished simulation this long after browser was closed
const timeout = 3 * time.Second

func keepBrowserAlive() {
	if time.Since(engine.KeepAlive()) < timeout {
		fmt.Println("keeping session open to browser")
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
			fmt.Println("openend web interface in", cmd)
			return
		}
	}
	fmt.Println("Please open ", url, " in a browser")
}

// list of browsers to try.
var browsers = []string{"x-www-browser", "google-chrome", "chromium-browser", "firefox", "ie", "iexplore"}
