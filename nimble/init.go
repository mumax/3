package nimble

// Initialization of general command line flags.

import (
	"code.google.com/p/mx3/core"
	"flag"
	"fmt"
	"io/ioutil"
	"log"
	"os"
	"os/exec"
	"runtime"
	"runtime/pprof"
	"time"
)

var (
	Flag_od       = flag.String("o", "", "set output directory")
	Flag_force    = flag.Bool("f", false, "force start, clean existing output directory")
	Flag_version  = flag.Bool("v", true, "print version")
	Flag_maxprocs = flag.Int("threads", 0, "maximum number of CPU threads, 0=auto")
	Flag_cpuprof  = flag.Bool("cpuprof", false, "Record gopprof CPU profile")
	//Flag_timing      = flag.Bool("timeprof", false, "Record timing profile")
	Flag_memprof     = flag.Bool("memprof", false, "Recored gopprof memory profile")
	Flag_debug       = flag.Bool("g", false, "Generate debug info")
	Flag_silent      = flag.Bool("s", false, "Don't generate any log info")
	Flag_verify      = flag.Bool("verify", true, "Verify crucial functionality")
	Flag_maxblocklen = flag.Int("maxblocklen", 1<<30, "Maximum size of concurrent blocks")
	Flag_minblocks   = flag.Int("minblocks", 1, "Minimum number of concurrent blocks")
	Flag_gpu         = flag.Int("gpu", 0, "specify GPU")
	Flag_sched       = flag.String("sched", "yield", "CUDA scheduling: auto|spin|yield|sync")
	Flag_pagelock    = flag.Bool("pagelock", true, "enable CUDA memeory page-locking")
	//Flag_hostkern    = flag.Bool("khost", false, "allocate convolution kernel in unified host memory")
)

var starttime time.Time

func Init() {
	flag.Parse()
	if *Flag_version {
		fmt.Print("Mumax Cubed 0.0 alpha ", runtime.GOOS, "_", runtime.GOARCH, " ", runtime.Version(), "(", runtime.Compiler, ")", "\n")
	}

	initOD()
	initLog()
	initGOMAXPROCS()
	initCpuProf()
	initMemProf()
	starttime = time.Now()
}

func initOD() {
	if *Flag_od != "" {
		core.SetOD(*Flag_od, *Flag_force)
	}
}

// SetOD sets the default output directory.
func SetOD(dir string) {
	core.SetOD(dir, *Flag_force)
}

func Cleanup() {
	core.Log("run time:", time.Since(starttime))
	core.Cleanup()
}

func initLog() {
	core.LOG = !*Flag_silent
	core.DEBUG = *Flag_debug
	log.SetPrefix(" ·")
	log.SetFlags(0)
}

func initGOMAXPROCS() {
	if *Flag_maxprocs == 0 {
		*Flag_maxprocs = runtime.NumCPU()
		core.Log("num CPU:", *Flag_maxprocs)
	}
	procs := runtime.GOMAXPROCS(*Flag_maxprocs) // sets it
	core.Log("GOMAXPROCS:", procs)
}

func initCpuProf() {
	if *Flag_cpuprof {
		// start CPU profile to file
		fname := core.OD + "/cpu.pprof"
		f, err := os.Create(fname)
		core.Fatal(err)
		core.Log("writing CPU profile to", fname)
		err = pprof.StartCPUProfile(f)
		core.Fatal(err)

		// at exit: exec go tool pprof to generate SVG output
		core.AtExit(func() {
			pprof.StopCPUProfile()
			me := procselfexe()
			outfile := fname + ".svg"
			core.SaveCmdOutput(outfile, "go", "tool", "pprof", "-svg", me, fname)
		})
	}
}

func initMemProf() {
	if *Flag_memprof {
		core.AtExit(func() {
			fname := core.OD + "/mem.pprof"
			f, err := os.Create(fname)
			defer f.Close()
			core.LogErr(err)
			core.Log("writing memory profile to", fname)
			core.LogErr(pprof.WriteHeapProfile(f))
			me := procselfexe()
			outfile := fname + ".svg"
			core.SaveCmdOutput(outfile, "go", "tool", "pprof", "-svg", "--inuse_objects", me, fname)
		})
	}
}

// path to the executable.
func procselfexe() string {
	me, err := os.Readlink("/proc/self/exe")
	core.PanicErr(err)
	return me
}
