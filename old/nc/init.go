package nc

import (
	"flag"
	"log"
	"os"
	"runtime"
	"runtime/pprof"
)

var (
	flag_version  = flag.Bool("V", false, "print version")
	flag_cuda     = flag.Bool("cuda", true, "use CUDA")
	flag_sched    = flag.String("yield", "auto", "CUDA scheduling: auto|spin|yield|sync")
	flag_pagelock = flag.Bool("pagelock", true, "enable CUDA memeory page-locking")
	flag_maxwarp  = flag.Int("warp", MAX_WARPLEN, "maximum elements per warp")
	flag_maxprocs = flag.Int("threads", 0, "maximum number of CPU threads, 0=auto")
	flag_cpuprof  = flag.String("cpuprof", "", "Write gopprof CPU profile to file")
	flag_memprof  = flag.String("memprof", "", "Write gopprof memory profile to file")
	flag_debug    = flag.Bool("debug", DEBUG, "Generate debug info")
	flag_log      = flag.Bool("log", LOG, "Generate log info")
)

func init() {
	flag.Parse()

	initLog()
	Debug("initializing")

	initGOMAXPROCS()
	initCpuProf()
	initMemProf()
	if *flag_version {
		PrintInfo(os.Stdout)
	}
	initWarp()
	initCUDA()
}

func initLog() {
	LOG = *flag_log
	DEBUG = *flag_debug
	log.SetFlags(log.Lmicroseconds)
	log.SetPrefix("#")
}

func initGOMAXPROCS() {
	if *flag_maxprocs == 0 {
		*flag_maxprocs = runtime.NumCPU()
	}
	procs := runtime.GOMAXPROCS(*flag_maxprocs) // sets it
	Log("GOMAXPROCS:", procs)
}

func initWarp() {
	MAX_WARPLEN = *flag_maxwarp
}

func initCpuProf() {
	if *flag_cpuprof != "" {
		f, err := os.Create(*flag_cpuprof)
		PanicErr(err)
		Log("Writing CPU profile to", *flag_cpuprof)
		err = pprof.StartCPUProfile(f)
		PanicErr(err)
		AtExit(pprof.StopCPUProfile)
	}
}
func initMemProf() {
	if *flag_memprof != "" {
		AtExit(func() {
			f, err := os.Create(*flag_memprof)
			defer f.Close()
			PanicErr(err)
			Log("Writing memory profile to", *flag_memprof)
			pprof.WriteHeapProfile(f)
		})
	}
}
