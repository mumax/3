package nimble

// Initialization of general command line flags.

import (
	"code.google.com/p/nimble-cube/core"
	"flag"
	"fmt"
	"log"
	"io/ioutil"
	"os"
	"os/exec"
	"runtime"
	"runtime/pprof"
	"time"
)

var (
	Flag_od          = flag.String("o", "", "set output directory")
	Flag_version     = flag.Bool("v", true, "print version")
	Flag_maxprocs    = flag.Int("threads", 0, "maximum number of CPU threads, 0=auto")
	Flag_cpuprof     = flag.Bool("cpuprof", false, "Record gopprof CPU profile")
	Flag_timing      = flag.Bool("timeprof", false, "Record timing profile")
	Flag_memprof     = flag.String("memprof", "", "Write gopprof memory profile to file")
	Flag_debug       = flag.Bool("g", false, "Generate debug info")
	Flag_silent      = flag.Bool("s", false, "Don't generate any log info")
	Flag_verify      = flag.Bool("verify", true, "Verify crucial functionality")
	Flag_maxblocklen = flag.Int("maxblocklen", 1<<30, "Maximum size of concurrent blocks")
	Flag_minblocks   = flag.Int("minblocks", 1, "Minimum number of concurrent blocks")
	Flag_gpu         = flag.Int("gpu", 0, "specify GPU")
	Flag_sched       = flag.String("sched", "yield", "CUDA scheduling: auto|spin|yield|sync")
	Flag_pagelock    = flag.Bool("pagelock", true, "enable CUDA memeory page-locking")
)

var starttime time.Time

func Init() {
	flag.Parse()
	if *Flag_version {
		fmt.Print("Nimble Cube 0.", core.BUILD_COMMIT, " alpha ", core.BUILD_DATE, " ", runtime.GOOS, "_", runtime.GOARCH, " ", runtime.Version(), "(", runtime.Compiler, ")", "\n")
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
		core.SetOD(*Flag_od)
	}
}

// SetOD sets the default output directory.
func SetOD(dir string) {
	core.SetOD(dir)
}

func Cleanup() {
	core.Log("run time:", time.Since(starttime))
	core.Cleanup()
}

func initLog() {
	core.LOG = !*Flag_silent
	core.DEBUG = *Flag_debug
	log.SetPrefix("#")
	log.SetFlags(log.Ltime)
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
		fname := core.OD + "/cpu.pprof"
 		f, err := os.Create(fname)
		core.PanicErr(err)
		core.Log("writing CPU profile to", fname)
		err = pprof.StartCPUProfile(f)
		core.PanicErr(err)
		core.AtExit(func(){
			pprof.StopCPUProfile()
			me, err := os.Readlink("/proc/self/exe")
			core.LogErr(err)
outfile := core.OD + "/cpu.pprof.svg"
			core.Log("exec:", "go", "tool", "pprof", "-svg", me, fname, ">", outfile)
			out, err := exec.Command("go", "tool", "pprof", "-svg", me, fname).Output()
			if err != nil {
				core.Log(string(out), err)
			}else{
				err := ioutil.WriteFile(outfile, out, 0666)
				core.LogErr(err)
			}
	})
}
}

func initMemProf() {
	if *Flag_memprof != "" {
		core.AtExit(func() {
			f, err := os.Create(*Flag_memprof)
			defer f.Close()
			core.PanicErr(err)
			core.Log("writing memory profile to", *Flag_memprof)
			pprof.WriteHeapProfile(f)
		})
	}
}
