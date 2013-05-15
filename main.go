package main

import (
	. "code.google.com/p/mx3/engine"
	"code.google.com/p/mx3/util"
	"flag"
	"log"
	"os/exec"
)

// dummy imports to fetch those files
import (
	_ "code.google.com/p/mx3/examples"
	_ "code.google.com/p/mx3/test"
)

func main() {
	Init()

	// flags parsed by engine.Init()
	switch flag.NArg() {
	case 1:
		if *Flag_od == "" { // -o not set
			SetOD(util.NoExt(flag.Arg(0))+".out", *Flag_force)
		}
		RunFile(flag.Arg(0))
	case 0:
		log.Println("no input files: starting interactive session")
		interactive()
	default:
		log.Fatal("need at most one input file")
	}

	keepBrowserAlive()
	Close()
}

// Enter interactive mode. Simulation is now exclusively controlled
// by web GUI (default: http://localhost:35367)
func RunInteractive() {
	lastKeepalive = time.Now()
	pause = true
	log.Println("entering interactive mode")
	if webPort == "" {
		goServe(*flag_port)
	}

	for {
		if time.Since(lastKeepalive) > webtimeout {
			log.Println("interactive session idle: exiting")
			break
		}
		log.Println("awaiting browser interaction")
		f := <-inject
		f()
	}
}

// Runs a script file.
func RunFile(fname string) {
	// first we compile the entire file into an executable tree
	f, err := os.Open(fname)
	util.FatalErr(err)
	defer f.Close()
	code, err2 := parser.Parse(f)
	util.FatalErr(err2)

	// now the parser is not used anymore so it can handle web requests
	web.goServe(*flag_port)

	// start executing the tree, possibly injecting commands from web gui
	for _, cmd := range code {
		cmd.Eval()
	}
}

// Compile file but do not run it. Used to check for errors.
func Vet(fname string) {
	f, err := os.Open(fname)
	util.FatalErr(err)
	defer f.Close()
	_, err = parser.Parse(f)
	util.FatalErr(err)
}

//
func interactive() {
	SetMesh(32, 32, 1, 5e-9, 5e-9, 5e-9)
	Msat = Const(1000e3)
	Aex = Const(10e-12)
	Alpha = Const(1)
	M.Set(Uniform(1, 1, 0))
	RunInteractive()
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
