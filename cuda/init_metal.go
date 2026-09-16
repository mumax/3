//go:build darwin && arm64

package cuda

import (
	"fmt"
	"log"
	"sync"

	"github.com/mumax/3/cuda/cu"
	"github.com/mumax/3/cuda/metal"
	"github.com/mumax/3/util"
)

var (
	Backend        = metal.Backend
	BackendVersion = metal.BackendVersion
	DriverVersion  = 0
	DevName        string
	TotalMem       int64
	GPUInfo        string
	Synchronous    bool
	UseCC          = 0
	cudaCtx        cu.Context
	metalInitMu    sync.Mutex
	metalReady     bool
)

// Init selects Apple's system-default Metal device and compiles the embedded
// MuMax3 MSL library. Apple Silicon exposes one process-wide unified-memory
// device, so non-zero CUDA-style device ordinals are ignored with a warning.
func Init(gpu int) {
	metalInitMu.Lock()
	defer metalInitMu.Unlock()
	if metalReady {
		return
	}
	if gpu != 0 {
		log.Printf(
			"Metal exposes the system-default Apple GPU; ignoring -gpu=%d",
			gpu,
		)
	}

	cu.Init(0)
	info, err := metal.Info()
	util.FatalErr(err)

	DevName = info.Name
	TotalMem = int64(info.RecommendedMaxWorkingSetSize)
	GPUInfo = fmt.Sprintf(
		"%s, unified memory, Metal %s, recommended working set %dMB",
		DevName,
		BackendVersion,
		TotalMem/(1024*1024),
	)
	cudaCtx = cu.Context(1)
	metalReady = true

	if Synchronous {
		log.Println("DEBUG: synchronized Metal calls")
	}
}

// stream0 is a source-compatible token for the one ordered Metal queue.
const stream0 = cu.Stream(0)

func Sync() {
	util.PanicErr(metal.Sync())
}
