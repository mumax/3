package mx

import (
	"fmt"
	"github.com/barnex/cuda5/cu"
	"runtime"
)

var cudaCtx cu.Context // gpu context to be used by all threads

func initGPU() {
	var flag uint
	switch *Flag_sched {
	default:
		panic("sched flag: expecting auto,spin,yield or sync: " + *Flag_sched)
	case "auto":
		flag = cu.CTX_SCHED_AUTO
	case "spin":
		flag = cu.CTX_SCHED_SPIN
	case "yield":
		flag = cu.CTX_SCHED_YIELD
	case "sync":
		flag = cu.CTX_BLOCKING_SYNC
	}
	tryCuInit()
	dev := cu.Device(*Flag_gpu)
	cudaCtx = cu.CtxCreate(flag, dev)
	M, m := dev.ComputeCapability()
	concurrent := dev.Attribute(cu.CONCURRENT_KERNELS)
	Log(fmt.Sprint("CUDA ", float32(cu.Version())/1000, " ",
		dev.Name(), "(", (dev.TotalMem())/(1024*1024), "MB) ",
		"compute ", M, ".", m,
		" concurrent: ", concurrent == 1))
	if M < 2 {
		Log("Compute capability does not allow unified addressing.")
		initStreamPool()
	}
}

// cu.Init(), but error is fatal and does not dump stack.
func tryCuInit() {
	defer func() {
		err := recover()
		FatalErr(err, "initialize GPU:")
	}()
	cu.Init(0)
}

// LockCudaThread locks the current goroutine to an OS thread
// and sets the CUDA context for that thread. To be called by
// every fresh goroutine that will use CUDA.
func LockCudaThread() {
	runtime.LockOSThread()
	cudaCtx.SetCurrent()
}
