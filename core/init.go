package core

// Initialization of general command line flags.

import (
	"flag"
	"fmt"
	"log"
	"os"
	"runtime"
	"runtime/pprof"
)

var (
	Flag_version     = flag.Bool("V", false, "print version")
	Flag_maxprocs    = flag.Int("threads", 0, "maximum number of CPU threads, 0=auto")
	Flag_cpuprof     = flag.String("cpuprof", "", "Write gopprof CPU profile to file")
	Flag_memprof     = flag.String("memprof", "", "Write gopprof memory profile to file")
	Flag_debug       = flag.Bool("debug", DEBUG, "Generate debug info")
	Flag_log         = flag.Bool("log", LOG, "Generate log info")
	Flag_verify      = flag.Bool("verify", true, "Verify crucial functionality")
	Flag_maxblocklen = flag.Int("maxblock", 1<<30, "Maximum size of concurrent blocks")
	// CUDA flags
	Flag_gpu      = flag.Int("gpu", 0, "specify GPU")
	Flag_sched    = flag.String("yield", "auto", "CUDA scheduling: auto|spin|yield|sync")
	Flag_pagelock = flag.Bool("pagelock", true, "enable CUDA memeory page-locking")

//	Flag_nantest   = flag.Bool("nantest", true, "Detect NaN/Inf early")
//	Flag_floattest = flag.Bool("floattest", true, "Detect float near-overflow")
)

func init() {
	flag.Parse()

	initLog()
	if *Flag_version {
		fmt.Println(log.Prefix(), "Nimble Cube", runtime.Version(), runtime.Compiler, runtime.GOOS, runtime.GOARCH)
	}
	initGOMAXPROCS()
	initCpuProf()
	initMemProf()
}

func initLog() {
	LOG = *Flag_log
	DEBUG = *Flag_debug
	log.SetPrefix("#")
}

func initGOMAXPROCS() {
	if *Flag_maxprocs == 0 {
		*Flag_maxprocs = runtime.NumCPU()
		Log("num CPU:", *Flag_maxprocs)
	}
	procs := runtime.GOMAXPROCS(*Flag_maxprocs) // sets it
	Log("GOMAXPROCS:", procs)
}

func initCpuProf() {
	if *Flag_cpuprof != "" {
		f, err := os.Create(*Flag_cpuprof)
		PanicErr(err)
		Log("writing CPU profile to", *Flag_cpuprof)
		err = pprof.StartCPUProfile(f)
		PanicErr(err)
		AtExit(pprof.StopCPUProfile)
	}
}

func initMemProf() {
	if *Flag_memprof != "" {
		AtExit(func() {
			f, err := os.Create(*Flag_memprof)
			defer f.Close()
			PanicErr(err)
			Log("writing memory profile to", *Flag_memprof)
			pprof.WriteHeapProfile(f)
		})
	}
}
