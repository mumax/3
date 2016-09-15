// mumax3 main command
package main

import (
	"flag"
	"fmt"
	"github.com/mumax/3/cuda"
	"github.com/mumax/3/engine"
	"github.com/mumax/3/prof"
	"github.com/mumax/3/script"
	"github.com/mumax/3/timer"
	"github.com/mumax/3/util"
	"log"
	"os"
	"runtime"
	"time"
)

var (
	flag_failfast = flag.Bool("failfast", false, "If one simulation fails, stop entire batch immediately")
	flag_test     = flag.Bool("test", false, "Cuda test (internal)")
	flag_version  = flag.Bool("v", true, "Print version")
	flag_vet      = flag.Bool("vet", false, "Check input files for errors, but don't run them")
	// more flags in engine/gofiles.go
)

func main() {
	flag.Parse()
	log.SetPrefix("")
	log.SetFlags(0)

	cuda.Init(*engine.Flag_gpu)
	runtime.GOMAXPROCS(runtime.NumCPU())
	cuda.Synchronous = *engine.Flag_sync
	if *flag_version {
		printVersion()
	}

	timer.Timeout = *engine.Flag_launchtimeout
	if *engine.Flag_launchtimeout != 0 {
		cuda.Synchronous = true
	}

	engine.TestDemag = *engine.Flag_selftest

	// used by bootstrap launcher to test cuda
	// successful exit means cuda was initialized fine
	if *flag_test {
		fmt.Println(cuda.GPUInfo)
		os.Exit(0)
	}

	engine.CacheDir = *engine.Flag_cachedir
	if *engine.Flag_cpuprof {
		prof.InitCPU(".")
	}
	if *engine.Flag_memprof {
		prof.InitMem(".")
	}
	defer prof.Cleanup()
	defer engine.Close() // flushes pending output, if any

	defer func() {
		if *engine.Flag_sync {
			timer.Print(os.Stdout)
		}
	}()

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
	fmt.Println("//no input files: starting interactive session")
	//initEngine()

	// setup outut dir
	now := time.Now()
	outdir := fmt.Sprintf("mumax-%v-%02d-%02d_%02dh%02d.out", now.Year(), int(now.Month()), now.Day(), now.Hour(), now.Minute())
	engine.InitIO(outdir, outdir, *engine.Flag_forceclean)

	engine.Timeout = 365 * 24 * time.Hour // basically forever

	// set up some sensible start configuration
	engine.Eval(`SetGridSize(128, 64, 1)
		SetCellSize(4e-9, 4e-9, 4e-9)
		Msat = 1e6
		Aex = 10e-12
		alpha = 1
		m = RandomMag()`)
	addr := goServeGUI()
	openbrowser("http://127.0.0.1" + addr)
	engine.RunInteractive()
}

// Runs a script file.
func runFileAndServe(fname string) {

	outDir := util.NoExt(fname) + ".out"
	if *engine.Flag_od != "" {
		outDir = *engine.Flag_od
	}
	engine.InitIO(fname, outDir, *engine.Flag_forceclean)

	fname = engine.InputFile

	var code *script.BlockStmt
	var err2 error
	if fname != "" {
		// first we compile the entire file into an executable tree
		code, err2 = engine.CompileFile(fname)
		util.FatalErr(err2)
	}

	// now the parser is not used anymore so it can handle web requests
	goServeGUI()

	if *engine.Flag_interactive {
		openbrowser("http://127.0.0.1" + *engine.Flag_port)
	}

	// start executing the tree, possibly injecting commands from web gui
	engine.EvalFile(code)

	if *engine.Flag_interactive {
		engine.RunInteractive()
	}
}

//func runRemote(fname string) {
//	URL, err := url.Parse(fname)
//	util.FatalErr(err)
//	host := URL.Host
//	engine.MountHTTPFS("http://" + host)
//	od := util.NoExt(URL.Path) + ".out"
//	engine.InitIO(od, *flag_force)
//	runFileAndServe(URL.Path) // TODO proxyserve?
//}

// start Gui server and return server address
func goServeGUI() string {
	if *engine.Flag_port == "" {
		log.Println(`//not starting GUI (-http="")`)
		return ""
	}
	addr := engine.GoServe(*engine.Flag_port)
	fmt.Print("//starting GUI at http://127.0.0.1", addr, "\n")
	return addr
}

// print version to stdout
func printVersion() {
	fmt.Print("//", engine.UNAME, "\n")
	fmt.Print("//", cuda.GPUInfo, ", using CC", cuda.UseCC, " PTX \n")
	fmt.Print("//(c) Arne Vansteenkiste, Dynamat LAB, Ghent University, Belgium", "\n")
	fmt.Print("//This is free software without any warranty. See license.txt", "\n")
}
