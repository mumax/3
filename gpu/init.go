package gpu

import (
	"code.google.com/p/nimble-cube/core"
	"code.google.com/p/nimble-cube/nimble"
	"flag"
	"fmt"
	"github.com/barnex/cuda5/cu"
	"runtime"
	"sync/atomic"
	"unsafe"
)

var cudaCtx cu.Context // gpu context to be used by all threads

func init() {
	flag.Parse()
	var flag uint
	switch *nimble.Flag_sched {
	default:
		panic("sched flag: expecting auto,spin,yield or sync: " + *nimble.Flag_sched)
	case "auto":
		flag = cu.CTX_SCHED_AUTO
	case "spin":
		flag = cu.CTX_SCHED_SPIN
	case "yield":
		flag = cu.CTX_SCHED_YIELD
	case "sync":
		flag = cu.CTX_BLOCKING_SYNC
	}
	cu.Init(0)
	dev := cu.Device(*nimble.Flag_gpu)
	cudaCtx = cu.CtxCreate(flag, dev)
	if *nimble.Flag_version {
		M, m := dev.ComputeCapability()
		fmt.Print("CUDA ", float32(cu.Version())/1000, " ", dev.Name(), " (", (dev.TotalMem())/(1024*1024), "MB", ", compute", M, ".", m, ")\n")
	}

}

var lockCount int32

// To be called by any fresh goroutine that will do cuda interaction.
func LockCudaThread() {
	if cudaCtx == 0 {
		return // allow to run if there's no GPU.
	}
	runtime.LockOSThread()
	cudaCtx.SetCurrent() // super cheap.
	c := atomic.AddInt32(&lockCount, 1)
	core.Debug("Locked thread", c, "to CUDA context")
}

// Undo LockCudaThread()
func UnlockCudaThread() {
	if cudaCtx == 0 {
		return // allow to run if there's no GPU.
	}
	runtime.UnlockOSThread()
	c := atomic.AddInt32(&lockCount, -1)
	core.Debug("Unlocked OS thread,", c, "remain locked")
}

// Register host memory for fast transfers,
// but only when flag -pagelock is true.
func MemHostRegister(slice []float32) {
	// do not fail on already registered memory.
	defer func() {
		err := recover()
		if err != nil && err != cu.ERROR_HOST_MEMORY_ALREADY_REGISTERED {
			panic(err)
		}
	}()
	if *nimble.Flag_pagelock {
		cu.MemHostRegister(unsafe.Pointer(&slice[0]), cu.SIZEOF_FLOAT32*int64(len(slice)), cu.MEMHOSTREGISTER_PORTABLE)
	}
}
