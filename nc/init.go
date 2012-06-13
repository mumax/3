package nc

import (
	"flag"
	"github.com/barnex/cuda4/cu"
	"log"
	"os"
	"runtime/pprof"
)

var (
	flag_version = flag.Bool("V", false, "print version")
	flag_maxwarp = flag.Int("warp", MAX_WARP, "maximum elements per warp")
	flag_sched   = flag.String("yield", "auto", "CUDA scheduling: auto|spin|yield|sync")
	flag_cpuprof = flag.String("cpuprof", "", "Write gopprof CPU profile to file")
	//flag_memprof    = flag.String("memprof", "", "Write gopprof memory profile to file")
)

var (
	cudaCtx cu.Context // gpu context to be used by all threads
)

func init() {
	flag.Parse()

	initCpuProf()

	if *flag_version {
		PrintInfo(os.Stdout)
	}

	initWarp()

	initCUDA()

	log.SetFlags(log.Lmicroseconds | log.Lshortfile)
	log.SetPrefix("#")
}

func initWarp() {
	MAX_WARP = *flag_maxwarp
	Log("max WarpLen:", MAX_WARP)
}

func initCUDA() {
	var flag uint
	switch *flag_sched {
	default:
		panic("sched flag: expecting auto,spin,yield or sync: " + *flag_sched)
	case "auto":
		flag = cu.CTX_SCHED_AUTO
	case "spin":
		flag = cu.CTX_SCHED_SPIN
	case "yield":
		flag = cu.CTX_SCHED_YIELD
	case "sync":
		flag = cu.CTX_BLOCKING_SYNC
	}
	Log("initializing CUDA")
	cu.Init(0)
	cudaCtx = cu.CtxCreate(flag, 0)
}

func initCpuProf() {
	if *flag_cpuprof != "" {
		f, err := os.Create(*flag_cpuprof)
		if err != nil {
			Log(err)
		}
		Log("Writing CPU profile to", *flag_cpuprof)
		pprof.StartCPUProfile(f)
		// TODO: flush!
	}
}
