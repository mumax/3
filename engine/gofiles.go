package engine

// support for running Go files as if they were mx3 files.

import (
	"flag"
	"github.com/mumax/3/cuda"
)

var (
	// These flags are shared between cmd/mumax3 and Go input files.
	Flag_cachedir      = flag.String("cache", "/tmp", "Kernel cache directory (empty disables caching)")
	Flag_cpuprof       = flag.Bool("cpuprof", false, "Record gopprof CPU profile")
	Flag_gpu           = flag.Int("gpu", 0, "Specify GPU")
	Flag_interactive   = flag.Bool("i", false, "Open interactive browser session")
	Flag_launchtimeout = flag.Duration("launchtimeout", 0, "Launch timeout for CUDA calls")
	Flag_memprof       = flag.Bool("memprof", false, "Recored gopprof memory profile")
	Flag_od            = flag.String("o", "", "Override output directory")
	Flag_port          = flag.String("http", ":35367", "Port to serve web gui")
	Flag_selftest      = flag.Bool("paranoid", false, "Enable convolution self-test for cuFFT sanity.")
	Flag_silent        = flag.Bool("s", false, "Silent") // provided for backwards compatibility
	Flag_sync          = flag.Bool("sync", false, "Synchronize all CUDA calls (debug)")
	Flag_forceclean    = flag.Bool("f", true, "Force start, clean existing output directory")
)

// Usage: in every Go input file, write:
//
// 	func main(){
// 		defer InitAndClose()()
// 		// ...
// 	}
//
// This initialises the GPU, output directory, etc,
// and makes sure pending output will get flushed.
func InitAndClose() func() {
	cuda.Init(0)
	InitIO("standardproblem4.mx3", "standardproblem4.out", true)
	GoServe(":35367")
	return func() {
		Close()
	}
}
