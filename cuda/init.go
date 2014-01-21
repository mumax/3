// Package cuda provides GPU interaction
package cuda

import (
	"fmt"
	"github.com/barnex/cuda5/cu"
	"github.com/mumax/3/util"
	"log"
	"runtime"
)

var (
	Version     float32    // cuda version
	DevName     string     // GPU name
	TotalMem    int64      // total GPU memory
	GPUInfo     string     // Human-readable GPU description
	Synchronous bool       // for debug: synchronize stream0 at every kernel launch
	cudaCtx     cu.Context // global CUDA context
	cudaCC      int        // compute capablity (used for fatbin)
)

// Locks to an OS thread and initializes CUDA for that thread.
func Init(gpu int) {
	if cudaCtx != 0 {
		return // needed for tests
	}

	runtime.LockOSThread()
	tryCuInit()
	dev := cu.Device(gpu)
	cudaCtx = cu.CtxCreate(cu.CTX_SCHED_YIELD, dev)
	cudaCtx.SetCurrent()

	M, m := dev.ComputeCapability()
	cudaCC = 10*M + m
	Version = float32(cu.Version()) / 1000
	DevName = dev.Name()
	TotalMem = dev.TotalMem()
	GPUInfo = fmt.Sprint("CUDA ", Version, " ", DevName, "(", (TotalMem)/(1024*1024), "MB) ", "cc", M, ".", m)

	if M < 2 {
		log.Fatalln("GPU has insufficient compute capability, need 2.0 or higher.")
	}
	if Synchronous {
		log.Println("DEBUG: synchronized CUDA calls")
	}

	// test PTX load so that we can catch CUDA_ERROR_NO_BINARY_FOR_GPU early
	fatbinLoad(madd2_map, "madd2")
}

// cu.Init(), but error is fatal and does not dump stack.
func tryCuInit() {
	defer func() {
		err := recover()
		util.FatalErr(err, "initialize GPU:")
	}()
	cu.Init(0)
}

// Global stream used for everything
const stream0 = cu.Stream(0)

// Synchronize the global stream
// (usually not needed, done automatically with -sync)
func Sync() {
	stream0.Synchronize()
}
