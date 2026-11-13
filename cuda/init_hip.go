//go:build hip

// Package cuda provides GPU interaction
package cuda

import (
	"fmt"
	"log"
	"runtime"

	"github.com/mumax/3/cuda/cu"
	"github.com/mumax/3/util"
)

var (
	DriverVersion int        // HIP driver version
	DevName       string     // GPU name
	TotalMem      int64      // total GPU memory
	GPUInfo       string     // Human-readable GPU description
	Synchronous   bool       // for debug: synchronize stream0 at every kernel launch
	cudaCtx       cu.Context // global HIP context
	cudaDev       int        // device ordinal
)

// Locks to an OS thread and initializes HIP for that thread.
func Init(gpu int) {
	if cudaCtx != 0 {
		return // needed for tests
	}

	runtime.LockOSThread()
	tryCuInit()
	cudaDev = gpu
	dev := cu.Device(gpu)
	cudaCtx = cu.CtxCreate(cu.CTX_SCHED_YIELD, dev)
	cudaCtx.SetCurrent()

	M, m := dev.ComputeCapability()
	DriverVersion = cu.Version()
	DevName = dev.Name()
	TotalMem = dev.TotalMem()
	GPUInfo = fmt.Sprintf("%s(%dMB), HIP Driver %d.%d, arch=%s",
		DevName, (TotalMem)/(1024*1024), DriverVersion/10000000, (DriverVersion%10000000)/100000, dev.ArchName())
	_ = M
	_ = m

	if Synchronous {
		log.Println("DEBUG: synchronized HIP calls")
	}

	// finalize one image at startup so a load/finalize failure is caught early
	fatbinLoad(madd2_image, "madd2")
}

// cu.Init(), but error is fatal and does not dump stack.
func tryCuInit() {
	defer func() {
		err := recover()
		util.FatalErr(err)
	}()
	cu.Init(0)
}

// Global stream used for everything
const stream0 = cu.Stream(0)

// Synchronize the global stream
// This is called before and after all memcopy operations between host and device.
func Sync() {
	stream0.Synchronize()
}
