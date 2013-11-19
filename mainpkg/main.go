package mainpkg

import (
	"flag"
	"fmt"
	"github.com/mumax/3/cuda"
	"github.com/mumax/3/engine"
	_ "github.com/mumax/3/ext"
	"github.com/mumax/3/prof"
	"github.com/mumax/3/util"
	"io/ioutil"
	"log"
	"runtime"
	"time"
)

var (
	flag_version = flag.Bool("v", false, "Print version")
	flag_silent  = flag.Bool("s", false, "Silent") // provided for backwards compatibility
	flag_vet     = flag.Bool("vet", false, "Check input files for errors, but don't run them")
	flag_od      = flag.String("o", "", "Override output directory")
	flag_force   = flag.Bool("f", false, "Force start, clean existing output directory")
	flag_port    = flag.String("http", ":35367", "Port to serve web gui")
	flag_cpuprof = flag.Bool("cpuprof", false, "Record gopprof CPU profile")
	flag_memprof = flag.Bool("memprof", false, "Recored gopprof memory profile")
	flag_gpu     = flag.Int("gpu", 0, "specify GPU")
	flag_sync    = flag.Bool("sync", false, "synchronize all CUDA calls (debug)")
	flag_time    = flag.Bool("time", false, "report walltime")
)

func Init() {

	if *flag_time {
		defer func() { log.Println("walltime:", time.Since(engine.StartTime)) }()
	}

	flag.Parse()
	//engine.DeclFunc("interactive", Interactive, "Wait for GUI interaction")

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
	}

	runtime.GOMAXPROCS(runtime.NumCPU())
	cuda.BlockSize = 512
	cuda.TileX = 32
	cuda.TileY = 32
	cuda.Init(*flag_gpu, "yield", *flag_sync)
	cuda.LockThread()

	if *flag_cpuprof {
		prof.InitCPU(engine.OD)
	}
	if *flag_memprof {
		prof.InitMem(engine.OD)
	}
	defer prof.Cleanup()

	engine.Init()
}

func RunFiles() {
	if flag.NArg() != 1 {
		batchMode()
		return
	}

	if *flag_od == "" { // -o not set
		engine.SetOD(util.NoExt(flag.Arg(0))+".out", *flag_force)
	}

	if !*flag_vet {
		fmt.Print("starting GUI at http://localhost", *flag_port, "\n")
		runFileAndServe(flag.Arg(0))
		keepBrowserAlive() // if open, that is
	}
}

// Runs a script file.
func runFileAndServe(fname string) {
	// first we compile the entire file into an executable tree
	bytes, err := ioutil.ReadFile(fname)
	util.FatalErr(err)
	code, err2 := engine.World.Compile(string(bytes))
	util.FatalErr(err2)

	// now the parser is not used anymore so it can handle web requests
	go engine.Serve(*flag_port)

	// start executing the tree, possibly injecting commands from web gui
	engine.EvalFile(code)
}
