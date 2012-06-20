package nc

import (
	"github.com/barnex/cuda4/cu"
	"runtime"
	"sync/atomic"
)

func initCUDA() {
	gomaxprocs = runtime.GOMAXPROCS(0)
	Debug("GOMAXPROCS:", gomaxprocs)

	// If we're not using CUDA, we should not ask CUDA to page-lock.
	if !*flag_cuda {
		*flag_pagelock = false
		return
	}

	Log("initializing CUDA")
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

var (
	cudaCtx    cu.Context // gpu context to be used by all threads
	gomaxprocs int
	cuprocs    int32
)

// To be called by any goroutine that wants to use cuda.
func LockCudaCtx() {
	if cudaCtx == 0 {
		return // allow to run if there's no GPU.
	}
	runtime.LockOSThread()
	ctx := cu.CtxGetCurrent()
	if ctx != cudaCtx {
		cudas := atomic.AddInt32(&cuprocs, 1)
		Debug("locking CUDA context:", "GOMAXPROCS:", gomaxprocs, ",CUDA-enabled threads:", cudas)
		cudaCtx.SetCurrent()
	}
}
