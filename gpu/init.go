package gpu

import (
	"flag"
	"github.com/barnex/cuda4/cu"
	"nimble-cube/core"
	"runtime"
	"sync/atomic"
	"unsafe"
)

var cudaCtx cu.Context // gpu context to be used by all threads

func init() {
	flag.Parse()
	var flag uint
	switch *core.Flag_sched {
	default:
		panic("sched flag: expecting auto,spin,yield or sync: " + *core.Flag_sched)
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
	core.Log("CUDA version:", cu.Version())
	dev := cu.Device(*core.Flag_gpu)
	cudaCtx = cu.CtxCreate(flag, dev)
	core.Log("GPU:", dev.Name(), (dev.TotalMem())/(1024*1024), "MB")
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
	if *core.Flag_pagelock {
		cu.MemHostRegister(unsafe.Pointer(&slice[0]), cu.SIZEOF_FLOAT32*int64(len(slice)), cu.MEMHOSTREGISTER_PORTABLE)
	}
}
