package main

import (
	"code.google.com/p/mx3/cuda"
	"code.google.com/p/mx3/engine"
	"code.google.com/p/mx3/util"
	"code.google.com/p/mx3/web"
	"flag"
	"io/ioutil"
	"log"
	"os/exec"
	"runtime"
	"time"
)

// dummy imports to fetch those files
import (
	_ "code.google.com/p/mx3/examples"
	_ "code.google.com/p/mx3/test"
)

var (
	flag_silent = flag.Bool("s", false, "Don't generate any log info")
	flag_od     = flag.String("o", "", "Override output directory")
	flag_force  = flag.Bool("f", false, "Force start, clean existing output directory")
	flag_port   = flag.String("http", ":35367", "Port to serve web gui")
)

func main() {

	flag.Parse()

	log.SetPrefix("")
	log.SetFlags(0)

	if flag.NArg() != 1 {
		log.Fatal("need one input file")
	}

	if *flag_silent {
		log.SetOutput(ioutil.Discard)
	}

	// TODO: tee output to log file, replace all panics by log.Panic

	if *flag_od == "" { // -o not set
		engine.SetOD(util.NoExt(flag.Arg(0))+".out", *flag_force)
	}

	log.Print(engine.UNAME, "\n")

	runtime.GOMAXPROCS(runtime.NumCPU())
	//TODO: init profiling // prof.Init(engine.OD)
	cuda.Init()
	cuda.LockThread()

	RunFileAndServe(flag.Arg(0))

	keepBrowserAlive() // if open, that is
	engine.Close()
}

// Runs a script file.
func RunFileAndServe(fname string) {
	// first we compile the entire file into an executable tree
	bytes, err := ioutil.ReadFile(fname)
	util.FatalErr(err)
	code, err2 := engine.Compile(string(bytes))
	util.FatalErr(err2)

	// now the parser is not used anymore so it can handle web requests
	web.GoServe(*flag_port)

	// start executing the tree, possibly injecting commands from web gui
	code.Exec()
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
