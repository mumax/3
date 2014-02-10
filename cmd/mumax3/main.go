package main

// mumax3 main command

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
	"runtime"
	"time"
)

var (
	flag_version     = flag.Bool("v", false, "Print version")
	flag_interactive = flag.Bool("i", false, "Open interactive browser session")
	flag_silent      = flag.Bool("s", false, "Silent") // provided for backwards compatibility
	flag_vet         = flag.Bool("vet", false, "Check input files for errors, but don't run them")
	flag_od          = flag.String("o", "", "Override output directory")
	flag_force       = flag.Bool("f", true, "Force start, clean existing output directory")
	flag_port        = flag.String("http", ":35367", "Port to serve web gui")
	flag_cpuprof     = flag.Bool("cpuprof", false, "Record gopprof CPU profile")
	flag_memprof     = flag.Bool("memprof", false, "Recored gopprof memory profile")
	flag_gpu         = flag.Int("gpu", 0, "specify GPU")
	flag_sync        = flag.Bool("sync", false, "synchronize all CUDA calls (debug)")
	flag_test        = flag.Bool("test", false, "cuda test (internal)")
)

func main() {
	flag.Parse()
	log.SetPrefix("")
	log.SetFlags(0)

	if *flag_version {
		printVersion()
	}

	cuda.Init(*flag_gpu)
	runtime.GOMAXPROCS(runtime.NumCPU())
	cuda.Synchronous = *flag_sync

	// used by bootstrap launcher to test cuda
	// successful exit means cuda was initialized fine
	if *flag_test {
		os.Exit(0)
	}

	if *flag_cpuprof {
		prof.InitCPU(engine.OD)
	}
	if *flag_memprof {
		prof.InitMem(engine.OD)
	}
	defer prof.Cleanup()
	defer engine.Close() // flushes pending output, if any

	if *flag_vet {
		vet()
		return
	}

	switch flag.NArg() {
	case 0:
		runInteractive()
	case 1:
		runFileAndServe(flag.Arg(0))
	default:
		RunQueue(flag.Args())
	}
}

func runInteractive() {
	fmt.Println("no input files: starting interactive session")
	//initEngine()

	// setup outut dir
	now := time.Now()
	outdir := fmt.Sprintf("mumax-%v-%02d-%02d_%02d:%02d.out", now.Year(), int(now.Month()), now.Day(), now.Hour(), now.Minute())
	suggestOD(outdir)

	engine.Timeout = 365 * 24 * time.Hour // forever

	// set up some sensible start configuration
	engine.Eval(`SetGridSize(128, 64, 1)
		SetCellSize(4e-9, 4e-9, 4e-9)
		Msat = 1e6
		Aex = 10e-12
		alpha = 1
		m = RandomMag()`)
	goServeGUI()
	engine.RunInteractive()
}

// Runs a script file.
func runFileAndServe(fname string) {
	//initEngine()
	suggestOD(util.NoExt(fname) + ".out")
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
	goServeGUI()

	if *flag_interactive {
		openbrowser("http://localhost" + *flag_port)
	}

	// start executing the tree, possibly injecting commands from web gui
	engine.EvalFile(code)

	if *flag_interactive {
		engine.RunInteractive()
	}
}

// start Gui server and return
func goServeGUI() {
	hostname, _ := os.Hostname()
	if hostname == "" {
		hostname = "localhost"
	}
	fmt.Print("starting GUI at http://", hostname, *flag_port, "\n")
	go engine.Serve(*flag_port)
}

// set output directory unless flag_od overrides it
func suggestOD(fname string) {
	if *flag_od != "" {
		fname = *flag_od
	}
	engine.SetOD(fname, *flag_force)
}

// print version to stdout
func printVersion() {
	fmt.Print("    ", engine.UNAME, "\n")
	fmt.Print("    ", cuda.GPUInfo, "\n")
	fmt.Print("(c) Arne Vansteenkiste, Dynamat LAB, Ghent University, Belgium", "\n")
	fmt.Print("    This is free software without any warranty. See license.txt", "\n")
	fmt.Print("\n")
}
