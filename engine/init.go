package engine

// File: initialization of command line flags.
// Author: Arne Vansteenkiste

import (
	"flag"
	"fmt"
	"log"
	"runtime"
	"time"
)

var (
	Flag_version     = flag.Bool("v", true, "print version")
	Flag_silent      = flag.Bool("s", false, "Don't generate any log info")
	Flag_od          = flag.String("o", "", "set output directory")
	Flag_force       = flag.Bool("f", false, "force start, clean existing output directory")
	Flag_maxprocs    = flag.Int("threads", 0, "maximum number of CPU threads, 0=auto")
	Flag_maxblocklen = flag.Int("maxblocklen", 1<<30, "Maximum size of concurrent blocks")
	Flag_minblocks   = flag.Int("minblocks", 1, "Minimum number of concurrent blocks")
)

func Init() {
	flag.Parse()

	initTiming()

	log.SetPrefix("")
	log.SetFlags(0)
	if *Flag_silent {
		log.SetOutput(devnul{})
	}

	if *Flag_version {
		log.Print("Mumax Cubed 0.0 alpha ", runtime.GOOS, "_", runtime.GOARCH, " ", runtime.Version(), "(", runtime.Compiler, ")", "\n")
	}

	if *Flag_od != "" {
		SetOD(*Flag_od, *Flag_force)
	}

	if *Flag_maxprocs == 0 {
		*Flag_maxprocs = runtime.NumCPU()
	}
	procs := runtime.GOMAXPROCS(*Flag_maxprocs) // sets it
	log.Println("gomaxprocs:", procs)
}

type devnul struct{}

func (d devnul) Write(b []byte) (int64, error) {
	return len(b), nil
}
