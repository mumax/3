package prof

import (
	"flag"
)

var (
	Flag_cpuprof = flag.Bool("cpuprof", false, "Record gopprof CPU profile")
	Flag_memprof = flag.Bool("memprof", false, "Recored gopprof memory profile")
	Flag_gpuprof = flag.Bool("gpuprof", false, "Recored GPU profile")
)

func Init() {
	initGPUProf() // should be before initGPU
	initGPU()
	initCpuProf()
	initMemProf()
}
