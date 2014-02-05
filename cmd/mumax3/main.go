package main

import (
	"flag"
	"fmt"
	"github.com/mumax/3/engine"
	. "github.com/mumax/3/init"
	"github.com/mumax/3/prof"
	"github.com/mumax/3/script"
	"github.com/mumax/3/util"
	"io/ioutil"
	"os/exec"
	"time"
)

func main() {

	Init()

	fname := flag.Arg(0)
	if fname == "" {
		now := time.Now()
		fname = fmt.Sprintf("mumax-%v-%02d-%02d_%02d:%02d.txt", now.Year(), int(now.Month()), now.Day(), now.Hour(), now.Minute())
	}
	if *Flag_od == "" { // -o not set
		engine.SetOD(util.NoExt(fname)+".out", *Flag_force)
	} else {
		engine.SetOD(*Flag_od, *Flag_force)
	}

	defer prof.Cleanup()
	defer engine.Close()
	RunFiles()
}

func RunFiles() {
	if !*Flag_vet {
		runFileAndServe(flag.Arg(0))
		keepBrowserAlive() // if open, that is
	}
}

// Runs a script file.
func runFileAndServe(fname string) {

	var code *script.BlockStmt
	var err2 error
	if fname != "" {
		// first we compile the entire file into an executable tree
		bytes, err := ioutil.ReadFile(fname)
		util.FatalErr(err)
		code, err2 = engine.World.Compile(string(bytes))
		util.FatalErr(err2)
	}

	// now the parser is not used anymore so it can handle web requests
	fmt.Print("starting GUI at http://localhost", *Flag_port, "\n")
	go engine.Serve(*Flag_port)

	if *Flag_interactive {
		openbrowser("http://localhost" + *Flag_port)
	}

	if fname != "" {
		// start executing the tree, possibly injecting commands from web gui
		engine.EvalFile(code)
	} else {
		fmt.Println("no input files: starting interactive session")
		engine.Timeout = 365 * 24 * time.Hour // forever
		// set up some sensible start configuration
		engine.Eval(`SetGridSize(128, 64, 1)
		SetCellSize(4e-9, 4e-9, 4e-9)
		Msat = 1e6
		Aex = 10e-12
		alpha = 1
		m = RandomMag()`)
		keepBrowserAlive()
	}
}

func keepBrowserAlive() {
	if time.Since(engine.KeepAlive()) < engine.Timeout {
		fmt.Println("keeping session open to browser")
		go func() {
			for {
				engine.Inject <- nop // wake up RunInteractive so it may exit
				time.Sleep(1 * time.Second)
			}
		}()
		engine.RunInteractive()
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
