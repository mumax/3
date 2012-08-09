package cli

import (
	"flag"
	"fmt"
	"log"
	"nimble-cube/core"
	"os"
	"runtime"
	"runtime/pprof"
)

var (
	Flag_version   = flag.Bool("V", false, "print version")
	Flag_maxprocs  = flag.Int("threads", 0, "maximum number of CPU threads, 0=auto")
	Flag_cpuprof   = flag.String("cpuprof", "", "Write gopprof CPU profile to file")
	Flag_memprof   = flag.String("memprof", "", "Write gopprof memory profile to file")
	Flag_debug     = flag.Bool("debug", core.DEBUG, "Generate debug info")
	Flag_log       = flag.Bool("log", core.LOG, "Generate log info")
	Flag_verify    = flag.Bool("verify", true, "Verify crucial functionality")
	Flag_nantest   = flag.Bool("nantest", true, "Detect NaN/Inf early")
	Flag_floattest = flag.Bool("floattest", true, "Detect float near-overflow")
)

var (
	DEFAULT_BUF = 16 // Buffer size for channels
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
	core.LOG = *Flag_log
	core.DEBUG = *Flag_debug
	log.SetPrefix("#")
}

func initGOMAXPROCS() {
	if *Flag_maxprocs == 0 {
		*Flag_maxprocs = runtime.NumCPU()
		core.Log("Num CPU:", *Flag_maxprocs)
	}
	procs := runtime.GOMAXPROCS(*Flag_maxprocs) // sets it
	core.Log("GOMAXPROCS:", procs)
}

func initCpuProf() {
	if *Flag_cpuprof != "" {
		f, err := os.Create(*Flag_cpuprof)
		core.PanicErr(err)
		core.Log("Writing CPU profile to", *Flag_cpuprof)
		err = pprof.StartCPUProfile(f)
		core.PanicErr(err)
		core.AtExit(pprof.StopCPUProfile)
	}
}

func initMemProf() {
	if *Flag_memprof != "" {
		core.AtExit(func() {
			f, err := os.Create(*Flag_memprof)
			defer f.Close()
			core.PanicErr(err)
			core.Log("Writing memory profile to", *Flag_memprof)
			pprof.WriteHeapProfile(f)
		})
	}
}
