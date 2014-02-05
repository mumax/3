// package init provides command-line flags and cuda initialization to both
// cmd/mumax3 and go input files.
package init

import (
	"flag"
	"fmt"
	"github.com/mumax/3/cuda"
	"github.com/mumax/3/engine"
	"github.com/mumax/3/prof"
	"log"
	"os"
	"runtime"
	"time"
)

var (
	Flag_version     = flag.Bool("v", false, "Print version")
	Flag_interactive = flag.Bool("i", false, "Open interactive browser session")
	Flag_silent      = flag.Bool("s", false, "Silent") // provided for backwards compatibility
	Flag_vet         = flag.Bool("vet", false, "Check input files for errors, but don't run them")
	Flag_od          = flag.String("o", "", "Override output directory")
	Flag_force       = flag.Bool("f", true, "Force start, clean existing output directory")
	Flag_port        = flag.String("http", ":35367", "Port to serve web gui")
	Flag_cpuprof     = flag.Bool("cpuprof", false, "Record gopprof CPU profile")
	Flag_memprof     = flag.Bool("memprof", false, "Recored gopprof memory profile")
	Flag_gpu         = flag.Int("gpu", 0, "specify GPU")
	Flag_sync        = flag.Bool("sync", false, "synchronize all CUDA calls (debug)")
	Flag_time        = flag.Bool("time", false, "report walltime")
	Flag_test        = flag.Bool("test", false, "cuda test (internal)")
)

func Init() {

	flag.Parse()

	if *Flag_time {
		defer func() { log.Println("walltime:", time.Since(engine.StartTime)) }()
	}

	log.SetPrefix("")
	log.SetFlags(0)

	if *Flag_version {
		fmt.Print("    ", engine.UNAME, "\n")
		fmt.Print("    ", cuda.GPUInfo, "\n")
		fmt.Print("(c) Arne Vansteenkiste, Dynamat LAB, Ghent University, Belgium", "\n")
		fmt.Print("    This is free software without any warranty. See license.txt", "\n")
		fmt.Print("\n")
	}

	runtime.GOMAXPROCS(runtime.NumCPU())
	cuda.BlockSize = 512
	cuda.TileX = 32
	cuda.TileY = 32
	cuda.Init(*Flag_gpu)
	cuda.Synchronous = *Flag_sync

	// used by bootstrap launcher to test cuda
	// successful exit means cuda was initialized fine
	if *Flag_test {
		os.Exit(0)
	}

	if *Flag_cpuprof {
		prof.InitCPU(engine.OD)
	}
	if *Flag_memprof {
		prof.InitMem(engine.OD)
	}

}
