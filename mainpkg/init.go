package mainpkg

import (
	"flag"
	"fmt"
	"github.com/mumax/3/cuda"
	"github.com/mumax/3/engine"
	"github.com/mumax/3/prof"
	"github.com/mumax/3/util"
	"log"
	"os"
	"runtime"
	"time"
)

var (
	flag_version     = flag.Bool("v", false, "Print version")
	flag_interactive = flag.Bool("i", false, "Open interactive browser session")
	flag_silent      = flag.Bool("s", false, "Silent") // provided for backwards compatibility
	flag_vet         = flag.Bool("vet", false, "Check input files for errors, but don't run them")
	flag_od          = flag.String("o", "", "Override output directory")
	flag_force       = flag.Bool("f", true, "Force start, clean existing output directory")
	flag_port        = flag.String("http", ":35367", "Port to serve web gui")
	flag_cpuprof     = flag.Bool("cpuprof", false, "Record gopprof CPU profile")
	flag_memprof     = flag.Bool("memprof", false, "Recored gopprof memory profile")
	flag_gpu         = flag.Int("gpu", 0, "specify GPU")
	flag_sync        = flag.Bool("sync", false, "synchronize all CUDA calls (debug)")
	flag_time        = flag.Bool("time", false, "report walltime")
	flag_test        = flag.Bool("test", false, "cuda test (internal)")
)

func Init() {

	flag.Parse()

	if *flag_time {
		defer func() { log.Println("walltime:", time.Since(engine.StartTime)) }()
	}

	log.SetPrefix("")
	log.SetFlags(0)

	if *flag_version {
		fmt.Print("    ", engine.UNAME, "\n")
		fmt.Print("    ", cuda.GPUInfo, "\n")
		fmt.Print("(c) Arne Vansteenkiste, Dynamat LAB, Ghent University, Belgium", "\n")
		fmt.Print("    This is free software without any warranty. See license.txt", "\n")
		fmt.Print("\n")
	}

	if *flag_vet {
		vet()
		return
	}

	runtime.GOMAXPROCS(runtime.NumCPU())
	cuda.BlockSize = 512
	cuda.TileX = 32
	cuda.TileY = 32
	cuda.Init(*flag_gpu, "yield", *flag_sync)
	cuda.LockThread()

	// used by bootstrap launcher to test cuda
	// successful exit means cuda was initialized fine
	if *flag_test {
		os.Exit(0)
	}

	if *flag_cpuprof {
		prof.InitCPU(engine.OD)
	}
	if *flag_memprof {
		prof.InitMem(engine.OD)
	}
	defer prof.Cleanup()

	fname := flag.Arg(0)
	if fname == "" {
		now := time.Now()
		fname = fmt.Sprintf("mumax-%v-%02d-%02d_%02d:%02d.txt", now.Year(), int(now.Month()), now.Day(), now.Hour(), now.Minute())
	}
	if *flag_od == "" { // -o not set
		engine.SetOD(util.NoExt(fname)+".out", *flag_force)
	} else {
		engine.SetOD(*flag_od, *flag_force)
	}

	engine.GUI.PrepareServer()
}
