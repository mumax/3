package nc

import (
	"github.com/barnex/cuda4/cu"
	"runtime"
	"sync"
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
	cuprocs    int
	culock     sync.Mutex
)

// Cheap function that assures the cuda context is set for this thread.
// To be called before any cuda function.
func SetCudaCtx() {
	if cudaCtx == 0 {
		return // allow to run if there's no GPU.
	}
	culock.Lock()
	defer culock.Unlock()
	if true { //cuprocs != gomaxprocs {
		//if cuprocs == gomaxprocs{Debug("what are the chances?")
		//return}
		ctx := cu.CtxGetCurrent()
		if ctx != cudaCtx {
			cuprocs++
			Debug("setting CUDA context", "gomaxprocs:", gomaxprocs, "cuprocs", cuprocs)
			cudaCtx.SetCurrent()
		}
	}
}
