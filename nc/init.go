package nc

import (
	"flag"
	"github.com/barnex/cuda4/cu"
	"log"
	"os"
)

var (
	flag_version         = flag.Bool("V", false, "print version")
	flag_maxwarp         = flag.Int("warp", MAX_WARP, "maximum elements per warp")
	flag_sched   *string = flag.String("sched", "auto", "CUDA scheduling: auto|spin|yield|sync")
)

var (
	context cu.Context // gpu context to be used by all threads
)

func init() {
	flag.Parse()

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

}
