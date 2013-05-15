package main

import (
	"code.google.com/p/mx3/engine"
	"code.google.com/p/mx3/util"
	"code.google.com/p/mx3/web"
	"flag"
	"log"
	"os"
	"os/exec"
	"time"
)

// dummy imports to fetch those files
import (
	_ "code.google.com/p/mx3/examples"
	_ "code.google.com/p/mx3/test"
)

func main() {
	// todo: move flags here
	engine.Init()

	if flag.NArg() == 1 {
		if *engine.Flag_od == "" { // -o not set
			engine.SetOD(util.NoExt(flag.Arg(0))+".out", *engine.Flag_force)
		}
		RunFileAndServe(flag.Arg(0))
	} else {
		log.Fatal("need one input file")
	}

	keepBrowserAlive() // if open, that is
	engine.Close()
}

// Enter interactive mode. Simulation is now exclusively controlled
// by web GUI (default: http://localhost:35367)
func RunInteractive() {
	web.LastKeepalive = time.Now()
	engine.Pause()
	log.Println("entering interactive mode")

	for {
		if time.Since(web.LastKeepalive) > web.Timeout {
			log.Println("interactive session idle: exiting")
			break
		}
		log.Println("awaiting browser interaction")
		f := <-engine.Inject
		f()
	}
}

// Runs a script file.
func RunFileAndServe(fname string) {
	// first we compile the entire file into an executable tree
	f, err := os.Open(fname)
	util.FatalErr(err)
	defer f.Close()
	code, err2 := engine.Compile(f)
	util.FatalErr(err2)

	// now the parser is not used anymore so it can handle web requests
	web.GoServe("")

	// start executing the tree, possibly injecting commands from web gui
	for _, cmd := range code {
		cmd.Eval()
	}
}

func keepBrowserAlive() {
	if time.Since(web.LastKeepalive) < web.Timeout {
		log.Println("keeping session open to browser")
		go func() {
			for {
				if time.Since(web.LastKeepalive) > web.Timeout {
					engine.Inject <- nop // wake up
				}
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
			log.Println("\n ====\n openend web interface in", cmd, "\n ====\n")
			return
		}
	}
	log.Println("\n ===== \n Please open ", url, " in a browser \n ==== \n")
}

// list of browsers to try.
var browsers = []string{"x-www-browser", "google-chrome", "chromium-browser", "firefox", "ie", "iexplore"}
