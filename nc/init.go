package nc

import (
	"flag"
	"log"
	"os"
)

var (
	flag_maxwarp = flag.Int("warp", MAX_WARP, "maximum elements per warp")
	flag_version = flag.Bool("V", false, "print version and exit")
)

func init() {
	flag.Parse()

	if *flag_version{
	PrintInfo(os.Stdout)
os.Exit(0)
	}

	MAX_WARP = *flag_maxwarp

	log.SetFlags(log.Lmicroseconds | log.Lshortfile)
	log.SetPrefix("#")
}
