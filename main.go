package main

import (
	"code.google.com/p/mx3/cuda"
	"code.google.com/p/mx3/engine"
	_ "code.google.com/p/mx3/ext"
	"code.google.com/p/mx3/prof"
	"code.google.com/p/mx3/util"
	"flag"
	"fmt"
	"io/ioutil"
	"log"
	"os"
	"os/exec"
	"runtime"
	"time"
)

var (
	flag_silent   = flag.Bool("s", false, "Don't generate any log info")
	flag_vet      = flag.Bool("vet", false, "Check input files for errors, but don't run them")
	flag_od       = flag.String("o", "", "Override output directory")
	flag_force    = flag.Bool("f", false, "Force start, clean existing output directory")
	flag_port     = flag.String("http", ":35367", "Port to serve web gui")
	flag_cpuprof  = flag.Bool("cpuprof", false, "Record gopprof CPU profile")
	flag_memprof  = flag.Bool("memprof", false, "Recored gopprof memory profile")
	flag_blocklen = flag.Int("bl", 512, "CUDA 1D thread block length")
	flag_blockX   = flag.Int("bx", 32, "CUDA 2D thread block size X")
	flag_blockY   = flag.Int("by", 32, "CUDA 2D thread block size Y")
)

func main() {
	start := time.Now()
	defer func() { log.Println("walltime:", time.Since(start)) }()

	flag.Parse()

	log.SetPrefix("")
	log.SetFlags(0)

	if *flag_vet {
		vet()
		return
	}

	if flag.NArg() != 1 {
		log.Fatal("need one input file")
	}

	if *flag_silent {
		log.SetOutput(ioutil.Discard)
	}

	log.Print(engine.UNAME, "\n")

	if *flag_od == "" { // -o not set
		engine.SetOD(util.NoExt(flag.Arg(0))+".out", *flag_force)
	}

	runtime.GOMAXPROCS(runtime.NumCPU())
	cuda.BlockSize = *flag_blocklen
	cuda.TileX = *flag_blockX
	cuda.TileY = *flag_blockY
	cuda.Init()
	cuda.LockThread()

	initProf()
	defer prof.Cleanup()

	RunFileAndServe(flag.Arg(0))

	keepBrowserAlive() // if open, that is
	engine.Close()
}

// Runs a script file.
func RunFileAndServe(fname string) {
	// first we compile the entire file into an executable tree
	bytes, err := ioutil.ReadFile(fname)
	util.FatalErr(err)
	code, err2 := engine.Compile(string(bytes))
	util.FatalErr(err2)

	// now the parser is not used anymore so it can handle web requests
	engine.GoServe(*flag_port)

	// start executing the tree, possibly injecting commands from web gui
	code.Eval()
}

// check all input files for errors, don't run.
func vet() {
	status := 0
	for _, f := range flag.Args() {
		src, ioerr := ioutil.ReadFile(f)
		util.FatalErr(ioerr)
		_, err := engine.Compile(string(src))
		if err != nil {
			fmt.Println(f, ":", err)
			status = 1
		} else {
			fmt.Println(f, ":", "OK")
		}
	}
	os.Exit(status)
}

// Enter interactive mode. Simulation is now exclusively controlled
// by web GUI (default: http://localhost:35367)
func RunInteractive() {
	//web.LastKeepalive = time.Now()
	engine.Pause()
	log.Println("entering interactive mode")
	panic("todo")
	//	for {
	//		if time.Since(web.LastKeepalive) > web.Timeout {
	//			log.Println("interactive session idle: exiting")
	//			break
	//		}
	//		log.Println("awaiting browser interaction")
	//		f := <-engine.Inject
	//		f()
	//	}
}

func keepBrowserAlive() {
	//	if time.Since(web.LastKeepalive) < web.Timeout {
	//		log.Println("keeping session open to browser")
	//		go func() {
	//			for {
	//				if time.Since(web.LastKeepalive) > web.Timeout {
	//					engine.Inject <- nop // wake up
	//				}
	//				time.Sleep(1 * time.Second)
	//			}
	//		}()
	//		RunInteractive()
	//	}
}

func nop() {}

// Try to open url in a browser. Instruct to do so if it fails.
func openbrowser(url string) {
	for _, cmd := range browsers {
		err := exec.Command(cmd, url).Start()
		if err == nil {
			log.Println("\n ====\n openend web interface in", cmd, "\n ====\n")
			return
		}
	}
	log.Println("\n ===== \n Please open ", url, " in a browser \n ==== \n")
}

// list of browsers to try.
var browsers = []string{"x-www-browser", "google-chrome", "chromium-browser", "firefox", "ie", "iexplore"}

func initProf() {
	if *flag_cpuprof {
		prof.InitCPU(engine.OD)
	}
	if *flag_memprof {
		prof.InitMem(engine.OD)
	}
}
