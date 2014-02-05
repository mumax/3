package main

import (
	"flag"
	"fmt"
	"github.com/mumax/3/cuda"
	"github.com/mumax/3/engine"
	"github.com/mumax/3/prof"
	"github.com/mumax/3/script"
	"github.com/mumax/3/util"
	"io/ioutil"
	"log"
	"os"
	"os/exec"
	"runtime"
	"time"
)

var (
	Flag_version     = flag.Bool("v", false, "Print version")
	Flag_interactive = flag.Bool("i", false, "Open interactive browser session")
	Flag_silent      = flag.Bool("s", false, "Silent") // provided for backwards compatibility
	Flag_vet         = flag.Bool("vet", false, "Check input files for errors, but don't run them")
	Flag_od          = flag.String("o", "", "Override output directory")
	Flag_force       = flag.Bool("f", true, "Force start, clean existing output directory")
	Flag_port        = flag.String("http", ":35367", "Port to serve web gui")
	Flag_cpuprof     = flag.Bool("cpuprof", false, "Record gopprof CPU profile")
	Flag_memprof     = flag.Bool("memprof", false, "Recored gopprof memory profile")
	Flag_gpu         = flag.Int("gpu", 0, "specify GPU")
	Flag_sync        = flag.Bool("sync", false, "synchronize all CUDA calls (debug)")
	Flag_test        = flag.Bool("test", false, "cuda test (internal)")
)

func main() {

	flag.Parse()

	log.SetPrefix("")
	log.SetFlags(0)

	if *Flag_version {
		fmt.Print("    ", engine.UNAME, "\n")
		fmt.Print("    ", cuda.GPUInfo, "\n")
		fmt.Print("(c) Arne Vansteenkiste, Dynamat LAB, Ghent University, Belgium", "\n")
		fmt.Print("    This is free software without any warranty. See license.txt", "\n")
		fmt.Print("\n")
	}

	engine.GUI.PrepareServer()

	runtime.GOMAXPROCS(runtime.NumCPU())
	cuda.Init(*Flag_gpu)
	cuda.Synchronous = *Flag_sync

	// used by bootstrap launcher to test cuda
	// successful exit means cuda was initialized fine
	if *Flag_test {
		os.Exit(0)
	}

	if *Flag_cpuprof {
		prof.InitCPU(engine.OD)
	}
	if *Flag_memprof {
		prof.InitMem(engine.OD)
	}

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
