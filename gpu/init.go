package gpu

import (
	"flag"
	"github.com/barnex/cuda4/cu"
	"nimble-cube/core"
	"runtime"
	"sync/atomic"
	"unsafe"
)

var (
	flag_sched    = flag.String("yield", "auto", "CUDA scheduling: auto|spin|yield|sync")
	flag_pagelock = flag.Bool("pagelock", true, "enable CUDA memeory page-locking")
)

var cudaCtx cu.Context // gpu context to be used by all threads

func init() {
	core.Log("initializing CUDA")
	var flag uint
	switch *flag_sched {
	default:
		panic("sched flag: expecting auto,spin,yield or sync: " + *flag_sched)
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
	cudaCtx = cu.CtxCreate(flag, 0)
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
	if *flag_pagelock {
		cu.MemHostRegister(unsafe.Pointer(&slice[0]), cu.SIZEOF_FLOAT32*int64(len(slice)), cu.MEMHOSTREGISTER_PORTABLE)
	}
}
