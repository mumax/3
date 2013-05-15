package engine

// File: initialization of command line flags.
// Author: Arne Vansteenkiste

import (
	"code.google.com/p/mx3/cuda"
	"code.google.com/p/mx3/prof"
	"flag"
	"io/ioutil"
	"log"
	"runtime"
)

var (
	flag_version  = flag.Bool("v", true, "print version")
	flag_silent   = flag.Bool("s", false, "Don't generate any log info")
	Flag_od       = flag.String("o", "", "set output directory")
	Flag_force    = flag.Bool("f", false, "force start, clean existing output directory")
	flag_maxprocs = flag.Int("threads", 0, "maximum number of CPU threads, 0=auto")
	flag_port     = flag.String("http", ":35367", "port to serve web gui")
)

const VERSION = "mx3.0.7 α "

var uname = VERSION + runtime.GOOS + "_" + runtime.GOARCH + " " + runtime.Version() + "(" + runtime.Compiler + ")"

// Initializes the simulation engine.
func Init() {
	flag.Parse()

	log.SetPrefix("")
	log.SetFlags(0)
	if *flag_silent {
		log.SetOutput(ioutil.Discard)
	}

	if *flag_version {
		log.Print(uname, "\n")
	}

	if *Flag_od != "" {
		SetOD(*Flag_od, *Flag_force)
	}

	if *flag_maxprocs == 0 {
		*flag_maxprocs = runtime.NumCPU()
	}
	procs := runtime.GOMAXPROCS(*flag_maxprocs) // sets it
	log.Println("gomaxprocs:", procs)

	prof.Init(OD)
	cuda.Init()
	cuda.LockThread()
}

// Cleanly exits the simulation, assuring all output is flushed.
func Close() {

	log.Println("shutting down")
	drainOutput()
	if Table != nil {
		Table.flush()
	}
	prof.Cleanup()
}
