package mx

// File: initialization of general command line flags.
// Author: Arne Vansteenkiste

import (
	"flag"
	"fmt"
	"runtime"
	"time"
)

var (
	Flag_version     = flag.Bool("v", true, "print version")
	Flag_debug       = flag.Bool("g", true, "Generate debug info")
	Flag_silent      = flag.Bool("s", false, "Don't generate any log info")
	Flag_od          = flag.String("o", "", "set output directory")
	Flag_force       = flag.Bool("f", false, "force start, clean existing output directory")
	Flag_cpuprof     = flag.Bool("cpuprof", false, "Record gopprof CPU profile")
	Flag_memprof     = flag.Bool("memprof", false, "Recored gopprof memory profile")
	Flag_maxprocs    = flag.Int("threads", 0, "maximum number of CPU threads, 0=auto")
	Flag_maxblocklen = flag.Int("maxblocklen", 1<<30, "Maximum size of concurrent blocks")
	Flag_minblocks   = flag.Int("minblocks", 1, "Minimum number of concurrent blocks")
	Flag_gpu         = flag.Int("gpu", 0, "specify GPU")
	Flag_sched       = flag.String("sched", "yield", "CUDA scheduling: auto|spin|yield|sync")
	Flag_pagelock    = flag.Bool("pagelock", true, "enable CUDA memeory page-locking")
)

func init() {
	flag.Parse()
	if *Flag_version && !*Flag_silent {
		fmt.Print("Mumax Cubed 0.0 alpha ", runtime.GOOS, "_", runtime.GOARCH, " ", runtime.Version(), "(", runtime.Compiler, ")", "\n")
	}

	initLog()
	initOD()
	initTiming()
	initGOMAXPROCS()
	initCpuProf()
	initMemProf()
}

func initTiming() {
	starttime := time.Now()
	AtExit(func() {
		Log("run time:", time.Since(starttime))
	})
}

func initGOMAXPROCS() {
	if *Flag_maxprocs == 0 {
		*Flag_maxprocs = runtime.NumCPU()
		Log("num CPU:", *Flag_maxprocs)
	}
	procs := runtime.GOMAXPROCS(*Flag_maxprocs) // sets it
	Log("GOMAXPROCS:", procs)
}
